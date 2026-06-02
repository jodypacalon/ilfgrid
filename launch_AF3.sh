#!/bin/bash

# Default parameter values
JOB_NAME="af3_job"
OUTPUT_DIR="$PWD/af3_output"
USE_GPU02=0
NUM_SEEDS=1

# Arrays holding the (repeatable) molecular entities
FASTA_FILES=()    # receptor FASTA files  (each may hold several records)
PEPTIDE_SEQS=()   # peptides: raw sequence strings OR FASTA file paths
LIGAND_SMILES=()  # ligands given as SMILES
CCD_CODES=()      # ligands given as CCD codes (e.g. ATP, HEM)

# Help function
show_help() {
    cat << EOF
Usage: ./submit_af3.sh [-f RECEPTOR.fasta] [-p PEPTIDE] [-l SMILES] [-c CCD] [OPTIONS]

A wrapper script to generate the AF3 JSON input and submit a Slurm job for a
multi-entity complex (any mix of receptors, peptides and ligands) on the ILF Grid.

All entity flags are REPEATABLE and can be mixed freely. You must supply at
least one entity. In AF3 a "receptor" and a "peptide" are both protein chains;
the two flags exist only for convenience.

Entity arguments (repeatable):
  -f, --fasta    FILE    Receptor FASTA file. A multi-record FASTA (several '>'
                         headers) is expanded into one protein chain per record.
  -p, --peptide  SEQ     Peptide chain. Accepts either a raw amino-acid sequence
                         string or a path to a FASTA file.
  -l, --ligand   SMILES  Ligand specified by SMILES string.
  -c, --ccd      CODE    Ligand specified by CCD code (e.g. ATP, HEM, NAD).

Optional arguments:
  -n, --name     NAME    Name of the job and output prefix (default: af3_job).
  -o, --output   DIR     Output directory (default: ./af3_output).
  -s, --seeds    N       Number of random seeds / predictions (default: 1).
  -g, --gpu02            Run specifically on gpu02 (general DBs + xla flash attention).
  -h, --help             Show this help message and exit.

Chain IDs are assigned automatically and uniquely (A, B, C, ... then AA, AB, ...)
across every protein and ligand in the order they are listed below: receptors
first, then peptides, then SMILES ligands, then CCD ligands.

Examples:
  # Two receptors + one peptide + one ligand, 5 seeds
  ./submit_af3.sh -f recA.fasta -f recB.fasta -p "GSHMKKLA..." -l "OC(=O)Cc1cn..." -n complex1 -s 5

  # Single multi-record FASTA (becomes several chains) + a cofactor by CCD code
  ./submit_af3.sh -f heterodimer.fasta -c ATP -n dimer_atp
EOF
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--fasta)   FASTA_FILES+=("$2");   shift ;;
        -p|--peptide) PEPTIDE_SEQS+=("$2");  shift ;;
        -l|--ligand)  LIGAND_SMILES+=("$2"); shift ;;
        -c|--ccd)     CCD_CODES+=("$2");     shift ;;
        -n|--name)    JOB_NAME="$2";         shift ;;
        -o|--output)  OUTPUT_DIR="$2";       shift ;;
        -s|--seeds)   NUM_SEEDS="$2";        shift ;;
        -g|--gpu02)   USE_GPU02=1 ;;
        -h|--help)    show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Validate that at least one entity was provided
if [ ${#FASTA_FILES[@]} -eq 0 ] && [ ${#PEPTIDE_SEQS[@]} -eq 0 ] \
   && [ ${#LIGAND_SMILES[@]} -eq 0 ] && [ ${#CCD_CODES[@]} -eq 0 ]; then
    echo "Error: provide at least one entity: -f (receptor), -p (peptide), -l (ligand SMILES) or -c (ligand CCD)."
    echo "Run './submit_af3.sh --help' for usage instructions."
    exit 1
fi

# Validate that every receptor FASTA file exists
for f in "${FASTA_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "Error: FASTA file '$f' not found."
        exit 1
    fi
done

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
JSON_FILE="${OUTPUT_DIR}/${JOB_NAME}.json"

# Write a small Python generator to a temp file and pass all data as argv.
# Passing values as argv (instead of interpolating into the Python source)
# avoids any quoting/escaping problems with SMILES strings.
PYGEN=$(mktemp "${TMPDIR:-/tmp}/af3_gen_XXXXXX.py")
trap 'rm -f "$PYGEN"' EXIT

cat << 'PYEOF' > "$PYGEN"
import json, sys, os, argparse, string


def chain_ids():
    """Yield unique IDs: A, B, ..., Z, AA, AB, ... (bijective base-26)."""
    letters = string.ascii_uppercase
    i = 0
    while True:
        n, s = i, ""
        while True:
            s = letters[n % 26] + s
            n = n // 26 - 1
            if n < 0:
                break
        yield s
        i += 1


def read_fasta(path):
    """Return a list of sequences (one per '>' record) from a FASTA file."""
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return seqs


p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--out", required=True)
p.add_argument("--seeds", type=int, required=True)
p.add_argument("--fasta", nargs="*", default=[])
p.add_argument("--peptide", nargs="*", default=[])
p.add_argument("--smiles", nargs="*", default=[])
p.add_argument("--ccd", nargs="*", default=[])
a = p.parse_args()

# Collect all protein chains (receptors first, then peptides)
protein_seqs = []

for fp in a.fasta:
    s = read_fasta(fp)
    if not s:
        print(f"Error: no sequence found in FASTA file '{fp}'.", file=sys.stderr)
        sys.exit(1)
    protein_seqs.extend(s)

for pep in a.peptide:
    if os.path.isfile(pep):                       # a FASTA path was given
        s = read_fasta(pep)
        if not s:
            print(f"Error: no sequence found in peptide FASTA '{pep}'.", file=sys.stderr)
            sys.exit(1)
        protein_seqs.extend(s)
    else:                                         # a raw sequence string was given
        protein_seqs.append(pep.strip())

# Build the sequences array with unique chain IDs
ids = chain_ids()
sequences = []
for seq in protein_seqs:
    sequences.append({"protein": {"id": next(ids), "sequence": seq}})
for smi in a.smiles:
    sequences.append({"ligand": {"id": next(ids), "smiles": smi}})
for code in a.ccd:
    sequences.append({"ligand": {"id": next(ids), "ccdCodes": [code]}})

if not sequences:
    print("Error: no entities to write.", file=sys.stderr)
    sys.exit(1)

data = {
    "name": a.name,
    "modelSeeds": list(range(1, a.seeds + 1)),
    "sequences": sequences,
    "dialect": "alphafold3",
    "version": 2,
}

with open(a.out, "w") as fh:
    json.dump(data, fh, indent=2)

n_prot = len(protein_seqs)
n_lig = len(a.smiles) + len(a.ccd)
print(f"Entities: {n_prot} protein chain(s), {n_lig} ligand(s)")
print(f"Chain IDs: {[list(e.values())[0]['id'] for e in sequences]}")
print(f"Seeds used: {list(range(1, a.seeds + 1))}")
PYEOF

# Generate the JSON. Empty arrays expand to nothing, which argparse nargs='*' handles.
python3 "$PYGEN" \
    --name "$JOB_NAME" \
    --out "$JSON_FILE" \
    --seeds "$NUM_SEEDS" \
    --fasta "${FASTA_FILES[@]}" \
    --peptide "${PEPTIDE_SEQS[@]}" \
    --smiles "${LIGAND_SMILES[@]}" \
    --ccd "${CCD_CODES[@]}"

if [ $? -ne 0 ]; then
    echo "Failed to generate JSON input."
    exit 1
fi

echo "Generated input JSON at: $JSON_FILE"

# Configure Slurm and AF3 arguments based on node selection
if [ "$USE_GPU02" -eq 1 ]; then
    SLURM_EXCLUDE=""
    SLURM_NODELIST="#SBATCH --nodelist=ilfgridgpun02fl"
    AF3_DB="/projects/ilfgrid/data/alphafold-genetic-databases"
    AF3_MODEL="/projects/ilfgrid/data/alphafold3_model_parameters"
    EXTRA_AF3_ARGS="--flash_attention_implementation xla"
else
    SLURM_EXCLUDE="#SBATCH --exclude=ilfgridgpun02fl"
    SLURM_NODELIST=""
    AF3_DB="/local_db/alphafold_db"
    AF3_MODEL="/local_db/alphafold3_model_parameters"
    EXTRA_AF3_ARGS=""
fi

SLURM_SCRIPT="${OUTPUT_DIR}/slurm_${JOB_NAME}.sh"

cat << EOF > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
$SLURM_EXCLUDE
$SLURM_NODELIST
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=${OUTPUT_DIR}/%x.%j.out
#SBATCH --job-name=${JOB_NAME}

module load miniconda/24.5.0

eval "\$(conda shell.bash hook)"
conda activate /projects/ilfgrid/apps/alphafold_v3.0.1/af301_conda_env

AF3_DIR="/projects/ilfgrid/apps/alphafold_v3.0.1"
AF3_DB="${AF3_DB}"
AF3_MODEL="${AF3_MODEL}"
INPUT="${JSON_FILE}"
OUTPUT="${OUTPUT_DIR}"

ENV_PYTHON="/projects/ilfgrid/apps/alphafold_v3.0.1/af301_conda_env/bin/python"

\$ENV_PYTHON \$AF3_DIR/run_alphafold.py --db_dir \$AF3_DB --json_path \$INPUT --output_dir \$OUTPUT --model_dir \$AF3_MODEL ${EXTRA_AF3_ARGS}
EOF

sbatch "$SLURM_SCRIPT"
echo "Submitted Slurm job for ${JOB_NAME} with ${NUM_SEEDS} seed(s)!"