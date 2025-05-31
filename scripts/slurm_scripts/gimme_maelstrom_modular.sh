#!/bin/bash
#SBATCH --job-name=gimme_maelstrom
#SBATCH --partition=cpu
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/gimme_maelstrom_%j.out
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/gimme_maelstrom_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org

###############################################################################
#                               CLI handling                                  #
###############################################################################
usage () {
    echo "Usage: sbatch gimme_maelstrom.sh [--input PATH] [--ref_genome STR] [--output_dir DIR] [--pfmfile STR]"
    echo
    echo "  --input        Path to the peaks/regions text file            (required)"
    echo "  --ref_genome   Reference genome name or FASTA path            (required)"
    echo "  --output_dir   Directory to write Maelstrom results           (required)"
    echo "  --pfmfile      Motif collection (name or .pfm file)           (optional)"
    exit 1
}

# --------- defaults (leave empty so we can detect missing args) --------------
input=""
ref_genome=""
output_dir=""
pfmfile="CisBP_ver2_Danio_rerio"   # default can still be overridden

# ----------------------------- parse flags -----------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input|-i)        input="$2";        shift 2 ;;
        --ref_genome|-r)   ref_genome="$2";   shift 2 ;;
        --output_dir|-o)   output_dir="$2";   shift 2 ;;
        --pfmfile|-p)      pfmfile="$2";      shift 2 ;;
        -h|--help)         usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# --------------------------- sanity checks -----------------------------------
[[ -z "$input" || -z "$ref_genome" || -z "$output_dir" ]] && \
    { echo "ERROR: --input, --ref_genome and --output_dir are mandatory."; usage; }

###############################################################################
#                         Environment & bookkeeping                           #
###############################################################################
module load mamba
mamba activate gimme

# Scratch-local cache to avoid SQLite lock contention
SCRATCH="/hpc/scratch/group.data.science/yangjoon.kim"
export XDG_CACHE_HOME="$SCRATCH/gimme_cache_${SLURM_JOB_ID}"
mkdir -p "$XDG_CACHE_HOME"

# If the user already has a GimmeMotifs cache, copy it over for speed
XDG_CACHE_HOME_ORIG="${XDG_CACHE_HOME_ORIG:-$HOME/.cache}"
if [[ -d "${XDG_CACHE_HOME_ORIG}/gimmemotifs" ]]; then
    cp -r "${XDG_CACHE_HOME_ORIG}/gimmemotifs" "$XDG_CACHE_HOME/"
fi
echo "Using XDG_CACHE_HOME=$XDG_CACHE_HOME"

# Ensure the output directory exists
mkdir -p "$output_dir"

###############################################################################
#                               Run Maelstrom                                 #
###############################################################################
echo "Running: gimme maelstrom $input $ref_genome $output_dir --pfmfile $pfmfile"
gimme maelstrom "$input" "$ref_genome" "$output_dir" --pfmfile "$pfmfile"
