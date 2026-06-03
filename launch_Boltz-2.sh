#!/bin/bash
###############################################################################
# run_boltz2.sh
#
# SLURM submission wrapper for Boltz-2 on the ILF grid (biowiki.sund.ku.dk).
#
# It (1) builds a Boltz-2 input YAML from command-line flags (proteins,
# peptides, ligands, optional affinity), (2) writes a matching SLURM batch
# script, and (3) submits it with sbatch. Use --dry-run to generate the files
# without submitting, or --yaml to submit your own hand-written YAML.
#
# Cluster facts baked in as defaults (see --help to override):
#   Boltz-2 install : /projects/ilfgrid/apps/Boltz-2
#   conda env       : /projects/ilfgrid/apps/Boltz-2/b2_env
#   module          : miniconda/24.5.0
#   GPU partition   : priority / qos prio / gpu:1 / exclude ilfgridgpun02fl
#   CPU partition   : cpu_jobs / qos cpu / 32 cpus
###############################################################################

set -euo pipefail

###############################################################################
# Defaults
###############################################################################
# --- Cluster / environment ---------------------------------------------------
CONDA_MODULE="miniconda/24.5.0"
CONDA_ENV="/projects/ilfgrid/apps/Boltz-2/b2_env"

# --- SLURM (gpu defaults; switched automatically for --accelerator cpu) ------
ACCELERATOR="gpu"          # gpu | cpu
JOB_NAME="boltz2"
PARTITION=""               # auto-selected from accelerator if left empty
QOS=""                     # auto-selected from accelerator if left empty
GPUS=1                     # number of GPUs (gpu mode only)
CPUS=""                    # auto: 4 (gpu) / 32 (cpu) if left empty
MEM="128G"
TIME="12:00:00"
NODES=1
EXCLUDE="ilfgridgpun02fl"  # GPU node to avoid (per wiki); cleared in cpu mode
EMAIL=""                   # if set, adds mail-user + mail-type=END,FAIL

# --- Boltz-2 prediction options ----------------------------------------------
OUT_DIR=""                 # default: ./<job_name>
USE_MSA_SERVER=1           # 1 = pass --use_msa_server (default on)
SINGLE_SEQUENCE=0          # 1 = msa: empty for all proteins (no MSA)
OVERRIDE=1                 # 1 = pass --override (default on, per wiki)
USE_POTENTIALS=0           # 1 = pass --use_potentials
OUTPUT_FORMAT="pdb"        # pdb | mmcif
DIFFUSION_SAMPLES=3        # NB: Boltz upstream default is 1; wiki examples use 3
RECYCLING_STEPS=3
SAMPLING_STEPS=200
STEP_SCALE=""              # empty -> Boltz default (1.638)
DIFFUSION_SAMPLES_AFFINITY=""   # empty -> Boltz default (5)
SAMPLING_STEPS_AFFINITY=""      # empty -> Boltz default (200)
AFFINITY_MW_CORRECTION=0
MAX_MSA_SEQS=""
SUBSAMPLE_MSA=0
NUM_SUBSAMPLED_MSA=""
NO_KERNELS=0               # set for older GPUs (cuequivariance issues)
WRITE_FULL_PAE=0
WRITE_FULL_PDE=0
NUM_WORKERS=""
PREPROC_THREADS=""
DEVICES=""                 # empty -> equals GPUS (gpu) or 1 (cpu)
MAX_PARALLEL_SAMPLES=""
CACHE=""                   # weights/data cache dir (Boltz default: ~/.boltz)
MSA_SERVER_URL=""
MSA_PAIRING_STRATEGY=""    # greedy | complete
METHOD=""
CHECKPOINT=""
AFFINITY_CHECKPOINT=""
EXTRA=""                   # raw passthrough appended verbatim to boltz predict

# --- Input building ----------------------------------------------------------
declare -a PROTEINS=()     # full-length proteins (GPCR, Galpha, ...)
declare -a PEPTIDES=()     # short protein chains (modeled as protein)
declare -a CCDS=()         # ligands by CCD code
declare -a SMILES=()       # ligands by SMILES string
AFFINITY_BINDER=""         # chain ID of the ligand to score affinity for
YAML_INPUT=""              # use this YAML directly, skip generation
SUBMIT=1                   # 0 with --dry-run

###############################################################################
# Help
###############################################################################
usage() {
cat <<'EOF'
run_boltz2.sh - SLURM wrapper to run Boltz-2 on the ILF grid

USAGE
  run_boltz2.sh [BUILD OPTIONS] [BOLTZ OPTIONS] [SLURM OPTIONS]
  run_boltz2.sh --yaml my_complex.yaml [BOLTZ OPTIONS] [SLURM OPTIONS]

The wrapper assembles a Boltz-2 YAML from the flags below, writes a SLURM
batch script next to it, and submits with sbatch. Sequences may be given
inline or read from a file with the @ prefix, e.g. --protein @gpcr.seq

INPUT / SYSTEM BUILDING
  -p, --protein SEQ|@FILE     Add a protein chain (repeatable). e.g. a GPCR or
                              a Galpha subunit. Use @file to read the sequence.
      --peptide  SEQ|@FILE    Add a peptide chain (repeatable). Modeled exactly
                              as a protein chain; flag is just for clarity.
  -s, --ligand-smiles SMILES  Add a ligand from a SMILES string (repeatable).
  -c, --ligand-ccd    CODE    Add a ligand from a CCD code, e.g. SAH (repeatable).
      --affinity CHAIN_ID     Compute binding affinity for this ligand chain
                              (must be a ligand chain; only ONE allowed). The
                              chain-ID map is printed after building the YAML.
      --yaml FILE             Submit this existing YAML directly; skip building.

  Chain IDs are auto-assigned A,B,C,... in this order:
      proteins -> peptides -> CCD ligands -> SMILES ligands

MSA
      --no-msa-server         Do NOT use the mmseqs2 MSA server (then proteins
                              need a precomputed MSA, or use --single-sequence).
      --single-sequence       Run proteins in single-sequence mode (msa: empty).
      --msa-server-url URL    Custom MSA server URL.
      --msa-pairing STRATEGY  greedy | complete.

BOLTZ PREDICTION OPTIONS
      --accelerator gpu|cpu   Compute device (default: gpu). Also selects the
                              SLURM partition/qos/resources automatically.
      --output-format FMT     pdb | mmcif (default: pdb).
      --diffusion-samples N   Number of structures to sample (default: 3).
      --recycling-steps N     Recycling steps (default: 3).
      --sampling-steps N      Diffusion sampling steps (default: 200).
      --step-scale FLOAT      Lower = more sample diversity (Boltz def 1.638).
      --use-potentials        Inference-time potentials (better physical poses).
      --no-override           Reuse cached preprocessing/predictions if present.
      --no-kernels            Disable cuequivariance kernels (older GPUs).
      --max-msa-seqs N        Max MSA sequences.
      --subsample-msa         Subsample the MSA.
      --num-subsampled-msa N  Number of MSA seqs to subsample.
      --write-full-pae        Save full PAE matrix.
      --write-full-pde        Save full PDE matrix.
      --num-workers N         Dataloader workers.
      --preproc-threads N     Preprocessing threads.
      --devices N             Devices for prediction (default: #GPUs / 1 on cpu).
      --max-parallel-samples N
      --cache DIR             Weights/data cache (Boltz default: ~/.boltz).
      --method STR            Boltz --method.
      --checkpoint PATH       Custom structure checkpoint.
      --affinity-checkpoint PATH
      --diffusion-samples-affinity N   (Boltz default: 5)
      --sampling-steps-affinity N      (Boltz default: 200)
      --affinity-mw-correction         Add molecular-weight correction.
      --extra "STR"           Raw string appended verbatim to `boltz predict`.

SLURM OPTIONS
  -J, --job-name NAME         Job name (default: boltz2). Also default out dir.
  -o, --out-dir DIR           Output / run directory (default: ./<job-name>).
      --partition NAME        Override SLURM partition.
      --qos NAME              Override SLURM qos.
      --gpus N                Number of GPUs (gpu mode, default: 1).
      --cpus N                CPUs per task (default: 4 gpu / 32 cpu).
      --mem SIZE              Memory (default: 128G).
      --time HH:MM:SS         Walltime (default: 12:00:00).
      --nodes N               Nodes (default: 1).
      --exclude LIST          Nodes to exclude (gpu default: ilfgridgpun02fl).
      --email ADDR            Send END/FAIL mail to this address.
      --conda-env PATH        Conda env (default: .../Boltz-2/b2_env).
      --conda-module NAME     Module to load (default: miniconda/24.5.0).

GENERAL
  -n, --dry-run               Build YAML + SLURM script but do NOT submit.
  -h, --help                  Show this help and exit.

EXAMPLES
  # GPCR + small molecule (SMILES) + peptide, score affinity for the ligand:
  run_boltz2.sh -J gpcr_lig_pep \
      --protein @gpcr.seq \
      --peptide  RVYIHPF \
      --ligand-smiles 'CC(=O)Oc1ccccc1C(=O)O' \
      --affinity C

  # GPCR + Galpha + peptide + a CCD ligand, more samples, mmcif output:
  run_boltz2.sh -J full_complex \
      --protein @gpcr.seq --protein @galpha.seq \
      --peptide @peptide.seq \
      --ligand-ccd SAH \
      --diffusion-samples 5 --output-format mmcif

  # Just build the files and inspect them first:
  run_boltz2.sh -J test --protein @gpcr.seq --ligand-smiles 'CCO' --dry-run

  # Submit a hand-written YAML on CPU:
  run_boltz2.sh --yaml my_complex.yaml --accelerator cpu -J cpu_run

NOTES
  * Affinity is protein--small-molecule only, ONE ligand binder, and works best
    for ligands of <= ~56 heavy atoms.
  * On GPU memory errors, retry with --accelerator cpu (slower but robust).
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

# Read a value that may be "@file" (slurp file contents, strip whitespace/newlines)
read_seq() {
  local v="$1"
  if [[ "$v" == @* ]]; then
    local f="${v:1}"
    [[ -f "$f" ]] || die "sequence file not found: $f"
    # join all non-empty, non-FASTA-header lines into one sequence string
    grep -v '^>' "$f" | tr -d '[:space:]'
  else
    echo "$v"
  fi
}

# Convert 0-based index to spreadsheet-style chain id: 0->A, 25->Z, 26->AA ...
chain_id() {
  local n=$1 s=""
  n=$((n + 1))
  while (( n > 0 )); do
    local r=$(((n - 1) % 26))
    s="$(printf "\\$(printf '%03o' $((65 + r)))")$s"
    n=$(((n - 1) / 26))
  done
  echo "$s"
}

###############################################################################
# Argument parsing
###############################################################################
[[ $# -eq 0 ]] && { usage; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--protein)        PROTEINS+=("$(read_seq "$2")"); shift 2 ;;
    --peptide)           PEPTIDES+=("$(read_seq "$2")"); shift 2 ;;
    -s|--ligand-smiles)  SMILES+=("$2"); shift 2 ;;
    -c|--ligand-ccd)     CCDS+=("$2"); shift 2 ;;
    --affinity)          AFFINITY_BINDER="$2"; shift 2 ;;
    --yaml)              YAML_INPUT="$2"; shift 2 ;;

    --no-msa-server)     USE_MSA_SERVER=0; shift ;;
    --single-sequence)   SINGLE_SEQUENCE=1; shift ;;
    --msa-server-url)    MSA_SERVER_URL="$2"; shift 2 ;;
    --msa-pairing)       MSA_PAIRING_STRATEGY="$2"; shift 2 ;;

    --accelerator)       ACCELERATOR="$2"; shift 2 ;;
    --output-format)     OUTPUT_FORMAT="$2"; shift 2 ;;
    --diffusion-samples) DIFFUSION_SAMPLES="$2"; shift 2 ;;
    --recycling-steps)   RECYCLING_STEPS="$2"; shift 2 ;;
    --sampling-steps)    SAMPLING_STEPS="$2"; shift 2 ;;
    --step-scale)        STEP_SCALE="$2"; shift 2 ;;
    --use-potentials)    USE_POTENTIALS=1; shift ;;
    --no-override)       OVERRIDE=0; shift ;;
    --no-kernels)        NO_KERNELS=1; shift ;;
    --max-msa-seqs)      MAX_MSA_SEQS="$2"; shift 2 ;;
    --subsample-msa)     SUBSAMPLE_MSA=1; shift ;;
    --num-subsampled-msa) NUM_SUBSAMPLED_MSA="$2"; shift 2 ;;
    --write-full-pae)    WRITE_FULL_PAE=1; shift ;;
    --write-full-pde)    WRITE_FULL_PDE=1; shift ;;
    --num-workers)       NUM_WORKERS="$2"; shift 2 ;;
    --preproc-threads)   PREPROC_THREADS="$2"; shift 2 ;;
    --devices)           DEVICES="$2"; shift 2 ;;
    --max-parallel-samples) MAX_PARALLEL_SAMPLES="$2"; shift 2 ;;
    --cache)             CACHE="$2"; shift 2 ;;
    --method)            METHOD="$2"; shift 2 ;;
    --checkpoint)        CHECKPOINT="$2"; shift 2 ;;
    --affinity-checkpoint) AFFINITY_CHECKPOINT="$2"; shift 2 ;;
    --diffusion-samples-affinity) DIFFUSION_SAMPLES_AFFINITY="$2"; shift 2 ;;
    --sampling-steps-affinity)    SAMPLING_STEPS_AFFINITY="$2"; shift 2 ;;
    --affinity-mw-correction)     AFFINITY_MW_CORRECTION=1; shift ;;
    --extra)             EXTRA="$2"; shift 2 ;;

    -J|--job-name)       JOB_NAME="$2"; shift 2 ;;
    -o|--out-dir)        OUT_DIR="$2"; shift 2 ;;
    --partition)         PARTITION="$2"; shift 2 ;;
    --qos)               QOS="$2"; shift 2 ;;
    --gpus)              GPUS="$2"; shift 2 ;;
    --cpus)              CPUS="$2"; shift 2 ;;
    --mem)               MEM="$2"; shift 2 ;;
    --time)              TIME="$2"; shift 2 ;;
    --nodes)             NODES="$2"; shift 2 ;;
    --exclude)           EXCLUDE="$2"; shift 2 ;;
    --email)             EMAIL="$2"; shift 2 ;;
    --conda-env)         CONDA_ENV="$2"; shift 2 ;;
    --conda-module)      CONDA_MODULE="$2"; shift 2 ;;

    -n|--dry-run)        SUBMIT=0; shift ;;
    -h|--help)           usage; exit 0 ;;
    *) die "unknown option: $1  (run with --help)" ;;
  esac
done

###############################################################################
# Validation & derived values
###############################################################################
[[ "$ACCELERATOR" == "gpu" || "$ACCELERATOR" == "cpu" ]] \
  || die "--accelerator must be gpu or cpu (got: $ACCELERATOR)"
[[ "$OUTPUT_FORMAT" == "pdb" || "$OUTPUT_FORMAT" == "mmcif" ]] \
  || die "--output-format must be pdb or mmcif"

# Partition / resource defaults by accelerator (overridable)
if [[ "$ACCELERATOR" == "gpu" ]]; then
  [[ -z "$PARTITION" ]] && PARTITION="standard"
  [[ -z "$QOS" ]]       && QOS="normal"
  [[ -z "$CPUS" ]]      && CPUS=4
  [[ -z "$DEVICES" ]]   && DEVICES="$GPUS"
else
  [[ -z "$PARTITION" ]] && PARTITION="cpu_jobs"
  [[ -z "$QOS" ]]       && QOS="cpu"
  [[ -z "$CPUS" ]]      && CPUS=32
  [[ -z "$DEVICES" ]]   && DEVICES=1
  EXCLUDE=""   # node exclusion is a GPU-node concern
fi

OUT_DIR="${OUT_DIR:-./$JOB_NAME}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"   # absolute

###############################################################################
# Build the YAML (unless one was supplied)
###############################################################################
if [[ -n "$YAML_INPUT" ]]; then
  [[ -f "$YAML_INPUT" ]] || die "--yaml file not found: $YAML_INPUT"
  YAML_PATH="$(cd "$(dirname "$YAML_INPUT")" && pwd)/$(basename "$YAML_INPUT")"
  echo "Using existing YAML: $YAML_PATH"
else
  (( ${#PROTEINS[@]} + ${#PEPTIDES[@]} + ${#CCDS[@]} + ${#SMILES[@]} > 0 )) \
    || die "no input given. Provide --protein/--peptide/--ligand-* or --yaml."

  YAML_PATH="$OUT_DIR/${JOB_NAME}.yaml"
  declare -a LIGAND_CHAINS=()   # track ligand chain IDs for affinity validation
  idx=0
  Y=""                          # YAML text accumulated here, written once at the end
  add() { Y+="$1"$'\n'; }       # append one line

  add "version: 1"
  add "sequences:"

  # --- proteins ---
  for seq in "${PROTEINS[@]}"; do
    cid="$(chain_id "$idx")"; idx=$((idx + 1))
    add "  - protein:"
    add "      id: $cid"
    add "      sequence: $seq"
    [[ "$SINGLE_SEQUENCE" -eq 1 ]] && add "      msa: empty"
    add "      # chain $cid: protein"
  done

  # --- peptides (modeled as protein) ---
  for seq in "${PEPTIDES[@]}"; do
    cid="$(chain_id "$idx")"; idx=$((idx + 1))
    add "  - protein:"
    add "      id: $cid"
    add "      sequence: $seq"
    [[ "$SINGLE_SEQUENCE" -eq 1 ]] && add "      msa: empty"
    add "      # chain $cid: peptide"
  done

  # --- CCD ligands ---
  for code in "${CCDS[@]}"; do
    cid="$(chain_id "$idx")"; idx=$((idx + 1))
    LIGAND_CHAINS+=("$cid")
    add "  - ligand:"
    add "      id: $cid"
    add "      ccd: $code"
  done

  # --- SMILES ligands ---
  for smi in "${SMILES[@]}"; do
    cid="$(chain_id "$idx")"; idx=$((idx + 1))
    LIGAND_CHAINS+=("$cid")
    add "  - ligand:"
    add "      id: $cid"
    add "      smiles: '$smi'"
  done

  # --- affinity ---
  if [[ -n "$AFFINITY_BINDER" ]]; then
    add "properties:"
    add "  - affinity:"
    add "      binder: $AFFINITY_BINDER"
  fi

  printf '%s' "$Y" > "$YAML_PATH"

  # Validate affinity binder is actually a ligand chain
  if [[ -n "$AFFINITY_BINDER" ]]; then
    ok=0
    for c in "${LIGAND_CHAINS[@]:-}"; do [[ "$c" == "$AFFINITY_BINDER" ]] && ok=1; done
    (( ok )) || die "--affinity '$AFFINITY_BINDER' is not a ligand chain. \
Ligand chains are: ${LIGAND_CHAINS[*]:-<none>}"
  fi

  echo "Generated YAML: $YAML_PATH"
  echo "----- chain map -----"
  grep -E '^\s+id:|# chain|ccd:|smiles:' "$YAML_PATH" | sed 's/^/  /'
  echo "---------------------"
fi

###############################################################################
# Assemble the boltz predict command
###############################################################################
BOLTZ_OPTS=(
  "--out_dir" "$OUT_DIR"
  "--accelerator" "$ACCELERATOR"
  "--output_format" "$OUTPUT_FORMAT"
  "--diffusion_samples" "$DIFFUSION_SAMPLES"
  "--recycling_steps" "$RECYCLING_STEPS"
  "--sampling_steps" "$SAMPLING_STEPS"
  "--devices" "$DEVICES"
)
(( USE_MSA_SERVER ))            && BOLTZ_OPTS+=("--use_msa_server")
(( OVERRIDE ))                  && BOLTZ_OPTS+=("--override")
(( USE_POTENTIALS ))            && BOLTZ_OPTS+=("--use_potentials")
(( NO_KERNELS ))                && BOLTZ_OPTS+=("--no_kernels")
(( SUBSAMPLE_MSA ))             && BOLTZ_OPTS+=("--subsample_msa")
(( WRITE_FULL_PAE ))            && BOLTZ_OPTS+=("--write_full_pae")
(( WRITE_FULL_PDE ))            && BOLTZ_OPTS+=("--write_full_pde")
(( AFFINITY_MW_CORRECTION ))    && BOLTZ_OPTS+=("--affinity_mw_correction")
[[ -n "$STEP_SCALE" ]]                  && BOLTZ_OPTS+=("--step_scale" "$STEP_SCALE")
[[ -n "$MAX_MSA_SEQS" ]]                && BOLTZ_OPTS+=("--max_msa_seqs" "$MAX_MSA_SEQS")
[[ -n "$NUM_SUBSAMPLED_MSA" ]]          && BOLTZ_OPTS+=("--num_subsampled_msa" "$NUM_SUBSAMPLED_MSA")
[[ -n "$NUM_WORKERS" ]]                 && BOLTZ_OPTS+=("--num_workers" "$NUM_WORKERS")
[[ -n "$PREPROC_THREADS" ]]             && BOLTZ_OPTS+=("--preprocessing-threads" "$PREPROC_THREADS")
[[ -n "$MAX_PARALLEL_SAMPLES" ]]        && BOLTZ_OPTS+=("--max_parallel_samples" "$MAX_PARALLEL_SAMPLES")
[[ -n "$CACHE" ]]                       && BOLTZ_OPTS+=("--cache" "$CACHE")
[[ -n "$MSA_SERVER_URL" ]]              && BOLTZ_OPTS+=("--msa_server_url" "$MSA_SERVER_URL")
[[ -n "$MSA_PAIRING_STRATEGY" ]]        && BOLTZ_OPTS+=("--msa_pairing_strategy" "$MSA_PAIRING_STRATEGY")
[[ -n "$METHOD" ]]                      && BOLTZ_OPTS+=("--method" "$METHOD")
[[ -n "$CHECKPOINT" ]]                  && BOLTZ_OPTS+=("--checkpoint" "$CHECKPOINT")
[[ -n "$AFFINITY_CHECKPOINT" ]]         && BOLTZ_OPTS+=("--affinity_checkpoint" "$AFFINITY_CHECKPOINT")
[[ -n "$DIFFUSION_SAMPLES_AFFINITY" ]]  && BOLTZ_OPTS+=("--diffusion_samples_affinity" "$DIFFUSION_SAMPLES_AFFINITY")
[[ -n "$SAMPLING_STEPS_AFFINITY" ]]     && BOLTZ_OPTS+=("--sampling_steps_affinity" "$SAMPLING_STEPS_AFFINITY")

BOLTZ_CMD="boltz predict \"$YAML_PATH\" ${BOLTZ_OPTS[*]} $EXTRA"

###############################################################################
# Write the SLURM batch script
###############################################################################
SLURM_SCRIPT="$OUT_DIR/${JOB_NAME}.slurm"
{
  echo "#!/bin/bash"
  echo "#SBATCH --job-name=$JOB_NAME"
  echo "#SBATCH --partition=$PARTITION"
  echo "#SBATCH --qos=$QOS"
  echo "#SBATCH --nodes=$NODES"
  echo "#SBATCH --ntasks-per-node=1"
  echo "#SBATCH --cpus-per-task=$CPUS"
  echo "#SBATCH --mem=$MEM"
  echo "#SBATCH --time=$TIME"
  [[ "$ACCELERATOR" == "gpu" ]] && echo "#SBATCH --gres=gpu:$GPUS"
  [[ -n "$EXCLUDE" ]]           && echo "#SBATCH --exclude=$EXCLUDE"
  echo "#SBATCH --output=$OUT_DIR/%x.%j.out"
  echo "#SBATCH --error=$OUT_DIR/%x.%j.err"
  if [[ -n "$EMAIL" ]]; then
    echo "#SBATCH --mail-user=$EMAIL"
    echo "#SBATCH --mail-type=END,FAIL"
  fi
  echo ""
  echo "set -euo pipefail"
  echo "echo \"Job \$SLURM_JOB_ID on \$(hostname) | start: \$(date)\""
  echo ""
  echo "module purge"
  echo "module load $CONDA_MODULE"
  echo "conda activate $CONDA_ENV"
  echo ""
  echo "$BOLTZ_CMD"
  echo ""
  echo "echo \"Finished: \$(date)\""
} > "$SLURM_SCRIPT"

echo "Generated SLURM script: $SLURM_SCRIPT"

###############################################################################
# Submit
###############################################################################
if (( SUBMIT )); then
  command -v sbatch >/dev/null 2>&1 || die "sbatch not found (are you on a login node?)"
  echo "Submitting..."
  JID="$(sbatch --parsable "$SLURM_SCRIPT")"
  echo "Submitted batch job $JID"
  echo "  monitor : squeue -j $JID"
  echo "  logs    : $OUT_DIR/${JOB_NAME}.${JID}.out"
  echo "  results : $OUT_DIR/predictions/"
else
  echo "[dry-run] Not submitting. Review the files above, then submit with:"
  echo "  sbatch \"$SLURM_SCRIPT\""
fi
