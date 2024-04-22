#!/bin/bash

#SBATCH --job-name=co_baseGRN      # Job name
#SBATCH --partition=cpu,gpu                     # Partition name
#SBATCH --output=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/co_baseGRN_%j.out # File to which STDOUT will be written, including job ID
#SBATCH --error=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/slurm_logs/co_baseGRN_%j.err  # File to which STDERR will be written, including job ID
#SBATCH --time=48:00:00                     # Runtime in HH:MM:SS            
#SBATCH --mem=64G                          # Memory total in GB (for all cores)
#SBATCH --cpus-per-task=1                  # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org  # Email to which notifications will be sent

# Usage check
if [ "$#" -ne 8 ]; then
    echo "Usage: sbatch $0 filepath peak_file CCAN_file cicero_score_threshold filename data_id save_figure figpath"
    exit 1
fi

# Assign command-line arguments to variables
# Define arguments for the first script
filepath=$1 # path to the peak file and CCAN file
peak_file=$2 # filename for the peak file
CCAN_file=$3 # filename for the CCAN file
cicero_score_threshold=$4 # Let's just use one annotation_class. (i.e. "global_annotation")
filename=$5 # filepath for the output
data_id=$6 # data_id for the output
save_figure=$7 # data_id for the output
figpath=$8 # filepath for the output

# Define arguments for the second script
#peaks_TSS_mapped=filename
# data_id=data_id
ref_genome="danRer11"
motif_score_threshold=10

# examples
# filepath="/path/to/your/data"
# peak_file="01_TDR118_CRG_arc_peaks.csv"
# CCAN_file="02_TDR118_cicero_connections_CRG_arc_peaks.csv"
# cicero_score_threshold=0.8
# filename="03_TDR118_processed_peak_file_danRer11.csv"
# data_id="TDR118"
# save_figure=True
# figpath="/path/to/your/figures"

# # Create the output directory if it doesn't exist
# mkdir -p "$output_filepath"

# gimmemotif parallelization workaround
$TMPDIR=/hpc/mydata/yang-joon.kim/
NEW_CACHE=$TMPDIR/cache_temp
mkdir -p $NEW_CACHE
if [ -z $XDG_CACHE_HOME ]; then
    XDG_CACHE_HOME=$HOME/.cache
fi
cp -r $XDG_CACHE_HOME/gimmemotifs $NEW_CACHE/
export XDG_CACHE_HOME=$NEW_CACHE
echo 'Using $XDG_CACHE_HOME for cache'

# Load necessary modules
module load anaconda
module load R/4.3
conda activate celloracle_env

# Navigate to the directory containing the python scripts
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/

# Run the first script with arguments
python run_03_celloracle_filter_CCANS_map_to_genes.py $filepath $peak_file $CCAN_file $cicero_score_threshold $filename $data_id $save_figure $figpath

# Run the second script with arguments
python run_04_celloracle_compute_baseGRN.py $filepath $filename $data_id $ref_genome $motif_score_threshold