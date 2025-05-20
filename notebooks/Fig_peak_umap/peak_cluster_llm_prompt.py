# A prompt for the LLM to query the identity of the peak clusters (from peak UMAP)
from litemind import GeminiApi
from litemind import OpenAIAPi
from litemind import combinedAPI
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
import os
import pdfkit
import markdown

# Initialize the Gemini API
api = combinedAPI()
# api = GeminiApi()

# Create the agent, passing the toolset
agent = Agent(api=api)

# Add a system message
agent.append_system_message("You are an expert developmental biologist specialized in vertebrate embryogenesis, especially zebrafish embryogenesis.")

# Function to write content to a markdown file
def write_to_markdown(content, filename="peak_clusters_report.md"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(content + "\n\n")
    return filename

# Create or clear the markdown file
with open("peak_clusters_report.md", "w", encoding="utf-8") as f:
    f.write("# Peak Clusters Report\n\n")

# Section 1: Project overview (description of the project)
# message = Message()
# message.append_text("Here is the manuscript of the project:\n")
# message.append_document('file:///Users/loic.royer/Desktop/applications/molecular_cartography/job_ad.pdf')
# message.append_text("Please analyze this manuscript, summarize the key objectives and data structures. list all all requirements and constraints, and then write a short summary of the job ad in your own words. \n")
# response = agent(message)
# write_to_markdown("## Manuscript Overview\n\n" + str(response))

# Manuscript overview
message = Message()
message.append_text("Here is the summary of the project:\n")
message.append_text("We are working on data from a study that generated a time-resolved single-cell multi-omic atlas of zebrafish embryogenesis by simultaneously profiling chromatin accessibility (scATAC-seq) and gene expression (scRNA-seq). Over 94,000 cells were sampled from pooled zebrafish embryos at six key developmental time points between 10 and 24 hours post-fertilization (hpf), covering the transition from gastrulation to early organogenesis. Single-nuclei dissociation was performed, followed by library preparation using the 10x Genomics Chromium Single Cell Multiome ATAC + Gene expression reagents, and sequencing was performed on the NovaSeq 6000 system. The resulting dataset includes paired chromatin accessibility (median 15,000 peaks per cell) and gene expression data (median 1,400 genes and 3,700 UMIs per cell) for each cell, allowing for integrated analysis of regulatory dynamics. \n")
response = agent(message)
write_to_markdown("## Manuscript Overview\n\n" + str(response))

# Project overview
message = Message()
message.append_text("I’m analyzing single-cell multiome datasets, which have both RNA and ATAC (chromatin accessibility) sequencing from the same cells, from whole zebrafish embryos during the six key developmental stages from 10 hours after fertilization to 24 hours after fertilization. \n")
message.append_text("I’m looking at the scATAC-seq dataset, which I pseudo-bulked using celltype and timepoint as two identifiers (“celltype & timepoint” as columns). The output, peaks-by-pseudobulk, is further processed for dimensionality reduction, UMAP, which gives the output")
# add the image (peak UMAP figure)
response = agent(message)
write_to_markdown("## Data Analysis Overview \n\n" + str(response))


# Section 2: Peak Clusters Analysis
message = Message()
message.append_text("The specific question that I have is to interpret/annotate each peak cluster from this. I’ll provide the following list of information as inputs from each peak cluster:"
                    "First, A dataframe of cluster-by-psuedobulk, whose columns are the “pseudobulk” (celltype x timepoint), averaged over the peaks in each peak cluster)\n"
                    "Second, A dataframe of cluster-by-genes, whose counts are the occurence of peaks associated with the genes. "
                    "Third,A dataframe of cluster-by-TF motifs, whose column is the list of TF motifs, and the counts are their enrichment scores (i.e., z-scores from GimmeMotifs maelstrom)"
                    "Lastly, a dataframe of motif-by-associated TFs, which is the database to link motifs to TFs\n\n")
message.append_text("\n\n \n")
#message.append_folder('/Users/loic.royer/Desktop/applications/molecular_cartography/candidates', allowed_extensions=['.pdf', 'docx', 'txt'], excluded_files=['.DS_Store'])
message.append_text("\n\n\n\nYour Analysis:\n")

response = agent(message)
write_to_markdown("## Peak Clusters Analysis\n\n" + str(response))

# Section 3: Summary of peak clusters
message = Message()
message.append_text("In the basis of your analysis of all peak clusters, please summarize the clusters based on shared biological pathways, cell types, developmental stages, and also the enriched Transcription Factors. \n")
response = agent(message)
write_to_markdown("## Peak Clusters Summary\n\n" + str(response))

