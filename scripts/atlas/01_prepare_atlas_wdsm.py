#!/usr/bin/env python3
"""Convert cached ATLAS bundles into Boltz-aligned WDSM NPZ datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

from genai_tps.utils.compute_device import (  # noqa: E402
    maybe_set_torch_cuda_current_device,
    parse_torch_device,
)

from genai_tps.backends.boltz.inference import build_boltz_inference_session  # noqa: E402
from genai_tps.backends.boltz.cache_paths import default_boltz_cache_dir  # noqa: E402
from genai_tps.data.atlas import cache_zip_path, read_ids_file  # noqa: E402
from genai_tps.data.atlas_convert import (  # noqa: E402
    concatenate_wdsm_arrays,
    convert_mdtraj_trajectory_to_wdsm,
    dump_structure_with_coordinate_ensemble,
    extract_atlas_zip,
    find_atlas_trajectory_files,
    load_mdtraj_frames,
    write_conversion_metadata,
    write_wdsm_npz,
)
from genai_tps.io.boltz_npz_export import load_topo  # noqa: E402
from genai_tps.training.dataset import split_wdsm_npz_train_val  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ATLAS protein MD frames as WDSM fine-tuning datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/atlas/raw"))
    parser.add_argument("--extract-dir", type=Path, default=Path("data/atlas/extracted"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/atlas/processed/pilot"))
    parser.add_argument("--boltz-processed-dir", type=Path, default=None,
                        help="Optional combined Boltz-processed output for multi-protein WDSM.")
    parser.add_argument("--yaml-dir", type=Path, default=None,
                        help="Directory containing one Boltz YAML per ID named {atlas_id}.yaml.")
    parser.add_argument("--yaml-map", type=Path, default=None,
                        help="Optional JSON mapping ATLAS ID to Boltz YAML path.")
    parser.add_argument("--topo-npz-dir", type=Path, default=None,
                        help="Optional directory with preprocessed Boltz {atlas_id}.npz topologies.")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device for Boltz preprocessing: cpu, cuda, or cuda:N.",
    )
    parser.add_argument("--diffusion-steps", type=int, default=1,
                        help="Only used when Boltz preprocessing is needed from YAML.")
    parser.add_argument("--recycling-steps", type=int, default=1,
                        help="Only used when Boltz preprocessing is needed from YAML.")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--max-frames-per-replicate", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite-extract", action="store_true")
    args = parser.parse_args()

    atlas_ids = read_ids_file(args.ids_file, limit=args.limit)
    yaml_map = _load_yaml_map(args.yaml_map)
    cache = Path(args.cache).expanduser() if args.cache else default_boltz_cache_dir()
    if str(args.device).strip().lower() == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = parse_torch_device(args.device)
        maybe_set_torch_cuda_current_device(device)
    combined_records = []
    frame_samples = []

    for atlas_id in atlas_ids:
        print(f"\n[prepare] {atlas_id}", flush=True)
        zip_path = cache_zip_path(args.raw_dir, atlas_id)
        extracted = extract_atlas_zip(
            zip_path,
            args.extract_dir / atlas_id,
            overwrite=args.overwrite_extract,
        )
        boltz_structure, topo_npz, source_processed_dir = _load_boltz_structure(
            atlas_id=atlas_id,
            args=args,
            yaml_map=yaml_map,
            cache=cache,
            device=device,
        )

        wdsm_chunks = []
        replicate_metadata = []
        for rep_idx, (replicate, xtc_path, topology_path) in enumerate(
            find_atlas_trajectory_files(extracted, atlas_id)
        ):
            print(f"[prepare]   replicate={replicate} xtc={xtc_path.name}", flush=True)
            traj = load_mdtraj_frames(
                xtc_path,
                topology_path,
                stride=args.stride,
                max_frames=args.max_frames_per_replicate,
                seed=args.seed + rep_idx,
            )
            wdsm = convert_mdtraj_trajectory_to_wdsm(traj, boltz_structure)
            wdsm_chunks.append(wdsm)
            replicate_metadata.append(
                {
                    "replicate": replicate,
                    "xtc_path": str(xtc_path),
                    "topology_path": str(topology_path),
                    "n_frames": int(wdsm.coords.shape[0]),
                    "n_atoms": int(wdsm.coords.shape[1]),
                    "stride": int(args.stride),
                    "max_frames_per_replicate": int(args.max_frames_per_replicate),
                }
            )

        coords, logw, atom_mask = concatenate_wdsm_arrays(wdsm_chunks)
        out_dir = args.processed_dir / atlas_id
        data_npz = out_dir / "training_dataset.npz"
        write_wdsm_npz(data_npz, coords=coords, logw=logw, atom_mask=atom_mask)
        write_conversion_metadata(
            out_dir / "atlas_conversion_metadata.json",
            {
                "atlas_id": atlas_id,
                "source_zip": str(zip_path),
                "n_samples": int(coords.shape[0]),
                "n_atoms": int(coords.shape[1]),
                "logw_policy": "uniform_zero_for_unbiased_md",
                "replicates": replicate_metadata,
            },
        )
        print(f"[prepare]   wrote {data_npz} shape={coords.shape}", flush=True)

        if args.val_fraction > 0:
            split_wdsm_npz_train_val(
                data_npz,
                out_dir / "dataset_split",
                val_fraction=args.val_fraction,
                seed=args.seed,
            )

        if args.boltz_processed_dir is not None:
            record_id, records = _write_boltz_processed_atlas_target(
                atlas_id=atlas_id,
                structure=boltz_structure,
                coords=coords,
                source_processed_dir=source_processed_dir,
                topo_npz=topo_npz,
                output_root=args.boltz_processed_dir,
            )
            combined_records.extend(records)
            frame_samples.extend(
                {"record_id": record_id, "frame_idx": idx, "logw": float(logw[idx])}
                for idx in range(len(logw))
            )

    if args.boltz_processed_dir is not None:
        _write_combined_manifest_and_frame_map(
            args.boltz_processed_dir,
            combined_records=combined_records,
            frame_samples=frame_samples,
            cache=cache,
        )


def _load_yaml_map(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): Path(v).expanduser() for k, v in payload.items()}


def _load_boltz_structure(*, atlas_id: str, args, yaml_map: dict[str, Path], cache: Path, device: torch.device):
    topo_npz = None
    source_processed_dir = None
    if args.topo_npz_dir is not None:
        candidate = Path(args.topo_npz_dir).expanduser() / f"{atlas_id}.npz"
        if candidate.is_file():
            topo_npz = candidate
    if topo_npz is None:
        yaml_path = _yaml_path_for_id(atlas_id, args.yaml_dir, yaml_map)
        bundle = build_boltz_inference_session(
            yaml_path=yaml_path,
            cache=cache,
            boltz_prep_dir=args.processed_dir / atlas_id / "boltz_prep",
            device=device,
            diffusion_steps=args.diffusion_steps,
            recycling_steps=args.recycling_steps,
            kernels=False,
        )
        topo_npz = bundle.topo_npz
        if topo_npz is None:
            raise RuntimeError(f"Boltz preprocessing did not produce a topology NPZ for {atlas_id}.")
        source_processed_dir = bundle.processed_dir
        del bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()
    structure, _n_struct = load_topo(Path(topo_npz))
    return structure, Path(topo_npz), source_processed_dir


def _yaml_path_for_id(atlas_id: str, yaml_dir: Path | None, yaml_map: dict[str, Path]) -> Path:
    if atlas_id in yaml_map:
        path = yaml_map[atlas_id]
    elif yaml_dir is not None:
        path = Path(yaml_dir).expanduser() / f"{atlas_id}.yaml"
    else:
        raise FileNotFoundError(
            "ATLAS preparation needs Boltz topology. Provide --topo-npz-dir, "
            "--yaml-dir, or --yaml-map."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Boltz YAML not found for {atlas_id}: {path}")
    return path


def _write_boltz_processed_atlas_target(
    *,
    atlas_id: str,
    structure,
    coords: np.ndarray,
    source_processed_dir: Path | None,
    topo_npz: Path,
    output_root: Path,
) -> tuple[str, list]:
    """Write one ATLAS target in Boltz-processed multi-protein layout."""
    from boltz.data.types import Manifest, Record  # noqa: PLC0415

    output_root = Path(output_root).expanduser()
    structures_dir = output_root / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    records = []
    record_id = atlas_id
    if source_processed_dir is not None and (source_processed_dir / "manifest.json").is_file():
        manifest = Manifest.load(source_processed_dir / "manifest.json")
        records = list(manifest.records)
        if records:
            record_id = records[0].id
    else:
        # The dataset can still consume this if the caller provides a matching manifest later.
        records = []

    dump_structure_with_coordinate_ensemble(
        structure,
        coords,
        structures_dir / f"{record_id}.npz",
    )

    if source_processed_dir is not None:
        for dirname in ("msa", "constraints", "templates", "mols"):
            src = source_processed_dir / dirname
            dst = output_root / dirname
            if not src.exists():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for path in src.iterdir():
                if path.is_file():
                    shutil.copy2(path, dst / path.name)
        if not records:
            raise RuntimeError(
                f"No manifest records found in {source_processed_dir}; cannot build combined manifest."
            )
    elif not records:
        raise RuntimeError(
            "--boltz-processed-dir requires YAML-based Boltz preprocessing so the combined "
            "manifest/MSA files can be copied. A bare --topo-npz-dir is insufficient."
        )

    print(
        f"[prepare]   wrote multi-protein StructureV2 {structures_dir / (record_id + '.npz')} "
        f"with {coords.shape[0]} frames",
        flush=True,
    )
    return record_id, records


def _write_combined_manifest_and_frame_map(
    output_root: Path,
    *,
    combined_records: list,
    frame_samples: list[dict[str, object]],
    cache: Path,
) -> None:
    from boltz.data.types import Manifest  # noqa: PLC0415

    output_root = Path(output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    Manifest(records=combined_records).dump(output_root / "manifest.json")
    (output_root / "frame_map.json").write_text(
        json.dumps({"samples": frame_samples}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    mol_src = Path(cache).expanduser() / "mols"
    mol_dst = output_root / "mols"
    if mol_src.is_dir() and not mol_dst.exists():
        try:
            mol_dst.symlink_to(mol_src, target_is_directory=True)
        except OSError:
            shutil.copytree(mol_src, mol_dst, dirs_exist_ok=True)
    print(f"[prepare] Combined manifest: {output_root / 'manifest.json'}", flush=True)
    print(f"[prepare] Combined frame map: {output_root / 'frame_map.json'}", flush=True)


if __name__ == "__main__":
    main()
