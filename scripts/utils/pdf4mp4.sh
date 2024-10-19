#!/bin/bash
# Script from Loic Royer (loic.royer@czbiohub.org) to convert a PDF file into a video

# Check for required arguments
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 <pdf_file> <seconds_per_double_page> [dpi]"
  exit 1
fi

PDF_FILE=$1
SECONDS_PER_PAGE=$2
DPI=${3:-600}  # Default DPI to 600 if not provided
OUTPUT_VIDEO="output.mp4"

# Step 1: Convert PDF to images with a white background and specified DPI
magick convert -density $DPI -background white -alpha remove -alpha off "$PDF_FILE" page_%04d.png

# Step 2: Combine images into double pages
i=0
while [ -f "page_$(printf "%04d" $i).png" ] && [ -f "page_$(printf "%04d" $(($i + 1))).png" ]; do
  magick convert +append "page_$(printf "%04d" $i).png" "page_$(printf "%04d" $(($i + 1))).png" "double_page_$(printf "%04d" $i).png"
  i=$(($i + 2))
done

# Step 3: Create input file for ffmpeg
> file_list.txt
for image in double_page_*.png; do
  echo "file '$image'" >> file_list.txt
  echo "duration $SECONDS_PER_PAGE" >> file_list.txt
done

# Add the last image without a duration to finalize the video
echo "file 'double_page_$(printf "%04d" $((i-2))).png'" >> file_list.txt

# Step 4: Create a video from double-page images with maximum quality
ffmpeg -y -f concat -safe 0 -i file_list.txt -vsync vfr -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p "$OUTPUT_VIDEO"

# Clean up intermediate files
rm page_*.png double_page_*.png file_list.txt

echo "Video created: $OUTPUT_VIDEO"