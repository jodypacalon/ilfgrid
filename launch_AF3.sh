#!/bin/bash

# Default parameter values
JOB_NAME="af3_job"
OUTPUT_DIR="$PWD/af3_output"
USE_GPU02=0
NUM_SEEDS=1

# Help function
show_help() {
    cat << EOF
Usage: ./submit_af3.sh -f RECEPTOR.fasta -l SMILES [OPTIONS]

A wrapper script to generate the AF3 JSON input and submit a Slurm job 
for a single-chain protein and ligand complex on the ILF Grid.

Required arguments:
  -f, --fasta   FILE        Path to the FASTA file containing the receptor sequence.
  -l, --ligand  SMILES      The SMILES string of the ligand.

Optional arguments:
  -n, --name    NAME        Name of the job and output prefix (default: af3_job).
  -o, --output  DIR         Output directory (default: ./af3_output).
  -s, --seeds   N           Number of random seeds / predictions (default: 1).
  -g, --gpu02               Run specifically on gpu02 (uses general DBs and xla flash attention).
  -h, --help                Show this help message and exit.

Example:
  ./submit_af3.sh -f receptor.fasta -l "OC(=O)Cc1cn..." -n "CamKIId_pipa" -s 5
EOF
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--fasta) FASTA_FILE="$2"; shift ;;
        -l|--ligand) LIGAND="$2"; shift ;;
        -n|--name) JOB_NAME="$2"; shift ;;
        -o|--output) OUTPUT_DIR="$2"; shift ;;
        -s|--seeds) NUM_SEEDS="$2"; shift ;;
        -g|--gpu02) USE_GPU02=1 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [ -z "$FASTA_FILE" ] || [ -z "$LIGAND" ]; then
    echo "Error: Both a FASTA file (-f) and ligand SMILES (-l) must be provided."
    echo "Run './submit_af3.sh --help' for usage instructions."
    exit 1
fi

# Validate FASTA file exists
if [ ! -f "$FASTA_FILE" ]; then
    echo "Error: FASTA file '$FASTA_FILE' not found."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
JSON_FILE="${OUTPUT_DIR}/${JOB_NAME}.json"

# Safely parse the FASTA and generate JSON using Python
python3 -c "
import json
import sys

fasta_path = '$FASTA_FILE'
smiles = r'$LIGAND'
job_name = '$JOB_NAME'
json_out = '$JSON_FILE'
num_seeds = int('$NUM_SEEDS')

sequence_lines = []
try:
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                sequence_lines.append(line)
except Exception as e:
    print(f'Error reading FASTA file: {e}')
    sys.exit(1)

protein_seq = ''.join(sequence_lines)

if not protein_seq:
    print('Error: No sequence found in the provided FASTA file.')
    sys.exit(1)

data = {
  'name': job_name,
  'modelSeeds': list(range(1, num_seeds + 1)),
  'sequences': [
    {
      'protein': {
        'id': 'A',
        'sequence': protein_seq
      }
    },
    {
      'ligand': {
        'id': 'LIG',
        'smiles': smiles
      }
    }
  ],
  'dialect': 'alphafold3',
  'version': 2
}

with open(json_out, 'w') as f:
    json.dump(data, f, indent=2)
print(f'Seeds used: {list(range(1, num_seeds + 1))}')
"

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