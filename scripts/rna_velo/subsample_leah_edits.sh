#!/bin/bash/
#This file will look through a single folder, find any samples with BAM files nested, and run velocyto on them
#Folder structure: Sample path > Sample(s) > BAM files, filtered_feature_bc_matrix > barcodes
#If you want to use your own barcodes, make a barcodes.tsv or barcodes.tsv.gz file, place it in the sample folder, and edit the barcode_path variable. 

#You can edit the paths and bash this file directly, or submit on the command line using
#bash Step1_RunVelocyto_10X.sh -w <working directory -s <sample path> -d <destination folder> -r <ref_path> -m <repeat mask> -b <path to barcodes tsv file> 

WDIR=/hpc/scratch/group.data.science/yangjoon.kim/zebrahub_multiome/rna_velo/
SAMPLE_PATH=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/00_CRG_arc_processed/
DESTDIR=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/06_rna_velo_processed/
GTF=/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11/genes/genes.gtf.gz

#note - copy this gtf file somewhere else, delete the last 3 lines or add "exon_number"
REPEAT_MASK=None
BARCODE_PATH=filtered_feature_bc_matrix/barcodes.tsv*

while getopts w:d:f:g:r: flag
do
    case "${flag}" in
        w) WDIR=${OPTARG};;
        d) DESTDIR=${OPTARG};;
        s) SAMPLE_PATH=${OPTARG};;
        r) REF_PATH=${OPTARG};;
        m) REPEAT_MASK=${OPTARG};;
	b) BARCODE_PATH=${OPTARG};;
    esac
done

#1. Make a working directory (if it doesn't exist)
[ -d "$WDIR" ] || mkdir -p "$WDIR"
cd "$WDIR"

[ -d "$DESTDIR" ] || mkdir -p "$DESTDIR"

#2. Set up directories for any scripts and slurm logs
[ -d "$DESTDIR/old_scripts" ] || mkdir -p "$DESTDIR/old_scripts"
[ -d "$DESTDIR/slurm_logs" ] || mkdir -p "$DESTDIR/slurm_logs"

#3. Grab the GTF file and copy into the working directory
if (file $GTF | grep -q compressed ) ; then
     cp $GTF annotation.gtf.gz
     gunzip annotation.gtf.gz
else
     cp $GTF annotation.gtf
fi

#allow a repeat mask or none
if test -f $REPEAT_MASK; then
  if (file $REPEAT_MASK | grep -q compressed ) ; then
     cp $REPEAT_MASK repeat_mask.gtf.gz
     gunzip repeat_mask.gtf.gz
  else
     cp $REPEAT_MASK repeat_mask.gtf
  fi
  MASK=Present
else
     echo "No Repeat Mask found, running unmasked. May take longer."
     MASK=None
fi


#3. Loop through the directory to find samples. Assume the directory contains multiple folders, each of which is one sample. 
for FOLDER in $SAMPLE_PATH/*; do
# pull out the sample name from the full path
FULLSAMPLE=${FOLDER##*/}
# this part is hard-coded to remove the _none_10x suffix from the sample name, this needs generalization
SAMPLE=${FULLSAMPLE%"_none_10x"}

echo "Running velocity on sample $SAMPLE"

#move any old contents into a different folder
if [ -d "$DESTDIR/$SAMPLE" ]; then
     echo "$DESTDIR/$SAMPLE already exists, moving into a subfolder"
     mkdir -p "$DESTDIR/old_files/"
	mv $DESTDIR/$SAMPLE/ $DESTDIR/old_files/
fi

#make the output folder
mkdir -p $DESTDIR/$SAMPLE/
OUTPUT_PATH=$DESTDIR/$SAMPLE/

#copy the bam files in here first, in case you don't have write permission in the source folder
cp -r $SAMPLE_PATH/$FULLSAMPLE/$SAMPLE/ $SAMPLE/



#create fraction variable
for fraction in {0.5,0.8,1}; do

#make an sbatch script to run velocity
cat > $SAMPLE-$fraction-velocity.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name $SAMPLE-$fraction-velocity # Name for your job
#SBATCH --ntasks 1              # Number of tasks
#SBATCH --cpus-per-task 32      # Number of cpus per task (64)
#SBATCH --time 60:00:00               # Runtime in hr:min:sec
#SBATCH --mem 256G            # Reserve RAM in gigabytes (G)
#SBATCH --partition cpu         # Partition to submit
#SBATCH --output $DESTDIR/slurm_logs/log-$SAMPLE-$fraction-velocity-%j.txt       # Standard out goes to this file
#SBATCH --error $DESTDIR/slurm_logs/log-$SAMPLE-$fraction-velocity-%j.txt        # Standard err goes to this file
#SBATCH --mail-user $USER@czbiohub.org     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc
#SBATCH --gpus 0            # Reserve 0 GPUs for usage


# RUN SCRIPT
module load data.science
source activate velocity
module load samtools

cd $SAMPLE/
mkdir fraction-$fraction/

for i in {1..5}; do
    echo "running iteration \$i"

    echo "filtering bam file"
    samtools view -h possorted_genome_bam.bam | awk -v frac="$fraction" 'BEGIN{srand();}{if(rand()<=frac || /^@/) print \$0;}' | samtools view -bS - > "fraction-$fraction/filtered-$SAMPLE-$fraction-\$i.bam"

    if [[ "$MASK" == "Present" ]]; then
        echo "Mask found, running velocity"
        velocyto run -e $SAMPLE-$fraction-\$i -b $BARCODE_PATH -o $OUTPUT_PATH -m ../repeat_mask.gtf fraction-$fraction/filtered-$SAMPLE-$fraction-\$i.bam ../annotation.gtf
    else
        echo "no mask file found, running unmasked"
        velocyto run -e $SAMPLE-$fraction-\$i -b $BARCODE_PATH -o $OUTPUT_PATH fraction-$fraction/filtered-$SAMPLE-$fraction-\$i.bam ../annotation.gtf

    fi
done

EOF

#now that the script is written, sbatch it and remove it
sbatch $SAMPLE-$fraction-velocity.sbatch
sleep 1
mv $SAMPLE-$fraction-velocity.sbatch $DESTDIR/old_scripts/
done
done