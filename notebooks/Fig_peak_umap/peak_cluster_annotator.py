# Peak Clusters Systematic Annotation System with Litemind
from litemind import combinedAPI
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.media.types.media_text import Text
from litemind.media.types.media_table import Table
import pandas as pd
import os
import numpy as np
from typing import Dict, List, Optional, Union
import json
import tempfile

class PeakClusterAnnotator:
    def __init__(self, api_key: Optional[str] = None, use_deep_research: bool = False):
        """Initialize the Peak Cluster Annotator with advanced litemind capabilities"""
        self.api = combinedAPI()
        
        # Configure model features for comprehensive analysis
        model_features = ["TextGeneration", "Tools", "Reasoning"]
        
        self.agent = Agent(
            api=self.api, 
            model_features=model_features
        )
        
        self.use_deep_research = use_deep_research
        
        # Enhanced system message for developmental biology expertise
        system_prompt = """You are an expert developmental biologist and computational genomics specialist with deep expertise in:

        **Zebrafish Embryogenesis (10-24 hpf):**
        - Gastrulation (10-11 hpf): Epiboly, involution, cell fate specification
        - Segmentation (11-16 hpf): Somitogenesis, neurulation, axis formation  
        - Early organogenesis (16-24 hpf): Neural tube formation, heart development, fin bud formation

        **Single-cell Multiome Analysis:**
        - scATAC-seq chromatin accessibility interpretation
        - Peak-to-gene regulatory relationships
        - Transcription factor binding motif analysis
        - Pseudobulk analysis and cell type deconvolution

        **Regulatory Networks:**
        - Master transcription factors (Sox, Tbx, Hox, etc.)
        - Developmental gene regulatory networks
        - Chromatin remodeling complexes
        - Enhancer-promoter interactions

        **Analysis Approach:**
        When analyzing peak clusters, systematically integrate:
        1. Temporal accessibility patterns (which developmental stages)
        2. Cell type specificity (which lineages/tissues)
        3. Transcription factor regulatory cascades
        4. Associated gene functions and pathways
        5. Developmental significance and biological interpretation

        Provide comprehensive, evidence-based annotations that connect molecular data to developmental biology mechanisms."""
        
        self.agent.append_system_message(system_prompt)
        
        # Initialize report file
        self.report_file = "peak_clusters_comprehensive_report.md"
        self.init_report()
    
    def init_report(self):
        """Initialize the markdown report file"""
        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write("# Peak Clusters Comprehensive Annotation Report\n\n")
            f.write("## Analysis Overview\n\n")
            f.write("This report contains systematic annotations of peak clusters from zebrafish ")
            f.write("scATAC-seq data during embryogenesis (10-24 hpf), analyzed using advanced ")
            f.write("multimodal AI capabilities.\n\n")
    
    def write_to_report(self, content: str):
        """Append content to the markdown report"""
        with open(self.report_file, "a", encoding="utf-8") as f:
            f.write(content + "\n\n")
    
    def create_table_media(self, df: pd.DataFrame, title: str, description: str) -> Table:
        """Convert pandas DataFrame to litemind Table media object"""
        # Save DataFrame to temporary CSV for litemind
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=True)
        temp_file.close()
        
        # Create Table media object
        table = Table(
            file_path=temp_file.name,
            title=title,
            description=description
        )
        
        return table, temp_file.name
    
    def create_research_query(self, cluster_id: str, cluster_data: Dict) -> str:
        """Create a sophisticated research query for deep analysis"""
        
        base_query = f"""
        ## Deep Research Query for Peak Cluster {cluster_id}
        
        **Primary Research Question:**
        What is the comprehensive biological identity and developmental significance of peak cluster {cluster_id} 
        in zebrafish embryogenesis (10-24 hpf)?
        
        **Specific Research Dimensions:**
        
        1. **Developmental Identity Analysis:**
           - Which specific developmental processes does this cluster regulate?
           - What are the key developmental stages where this cluster is most active?
           - Which cell lineages/tissues show the strongest chromatin accessibility?
        
        2. **Regulatory Network Characterization:**
           - What are the master transcription factors controlling this cluster?
           - How do these TFs connect to known developmental gene regulatory networks?
           - What are the predicted regulatory cascades and feedback loops?
        
        3. **Functional Pathway Analysis:**
           - Which biological pathways and processes are enriched in associated genes?
           - How do these pathways relate to zebrafish embryonic development?
           - What are the predicted phenotypic consequences of disrupting this cluster?
        
        4. **Comparative Developmental Analysis:**
           - How does this cluster compare to known developmental regulatory modules?
           - What are similar regulatory programs in other model organisms?
           - How does this fit into the broader zebrafish developmental timeline?
        
        5. **Experimental Validation Hypotheses:**
           - What specific experiments would validate the predicted functions?
           - Which genes/TFs would be priority targets for functional studies?
           - What phenotypes would be expected from perturbation experiments?
        
        **Analysis Integration Requirements:**
        - Synthesize chromatin accessibility, gene associations, and TF motif data
        - Connect molecular patterns to developmental biology mechanisms
        - Provide specific, testable hypotheses about cluster function
        - Contextualize within zebrafish embryogenesis timeline and cell fate decisions
        """
        
        return base_query
    
    def annotate_single_cluster(self, cluster_id: str, cluster_data: Dict) -> str:
        """Annotate a single peak cluster using advanced litemind capabilities"""
        
        # Create the research message
        message = Message(role="user")
        
        # Add the research query or standard analysis request
        if self.use_deep_research:
            query_text = self.create_research_query(cluster_id, cluster_data)
        else:
            query_text = f"""
            ## Peak Cluster {cluster_id} Comprehensive Analysis
            
            Please provide a detailed biological annotation for this peak cluster from our zebrafish 
            embryogenesis scATAC-seq dataset. Focus on:
            
            1. **Biological Identity**: Cell type(s) and developmental process(es)
            2. **Temporal Dynamics**: Activity patterns across 10-24 hpf stages  
            3. **Regulatory Control**: Key transcription factors and regulatory mechanisms
            4. **Functional Significance**: Biological pathways and developmental roles
            5. **Integration**: How this cluster fits into zebrafish embryogenesis
            
            Provide a comprehensive analysis (600-1000 words) suitable for a research publication.
            """
        
        message.append_text(query_text)
        
        # Add cluster metadata
        metadata_text = f"""
        ### Cluster Metadata
        - **Cluster ID**: {cluster_data.get('cluster_id', 'Unknown')}
        - **Number of peaks**: {cluster_data.get('n_peaks', 'Unknown')}
        - **Cluster level**: {cluster_data.get('cluster_level', 'Unknown')}
        - **Parent cluster**: {cluster_data.get('parent_cluster', 'None')}
        """
        message.append_text(metadata_text)
        
        # Add dataframes as tables using litemind's Table media
        temp_files = []  # Keep track of temp files for cleanup
        
        try:
            # Chromatin accessibility data
            if 'pseudobulk_accessibility' in cluster_data:
                df = cluster_data['pseudobulk_accessibility']
                table, temp_file = self.create_table_media(
                    df, 
                    f"Cluster {cluster_id} Chromatin Accessibility",
                    "Pseudobulk chromatin accessibility scores across cell types and developmental timepoints"
                )
                message.append_table(table)
                temp_files.append(temp_file)
            
            # Associated genes
            if 'associated_genes' in cluster_data:
                df = cluster_data['associated_genes']
                table, temp_file = self.create_table_media(
                    df,
                    f"Cluster {cluster_id} Associated Genes", 
                    "Genes associated with peaks in this cluster through peak-to-gene linkage"
                )
                message.append_table(table)
                temp_files.append(temp_file)
            
            # TF motifs
            if 'tf_motifs' in cluster_data:
                df = cluster_data['tf_motifs']
                table, temp_file = self.create_table_media(
                    df,
                    f"Cluster {cluster_id} TF Motifs",
                    "Enriched transcription factor motifs with z-scores from motif analysis"
                )
                message.append_table(table)
                temp_files.append(temp_file)
            
            # Motif-to-TF mapping
            if 'motif_to_tf_mapping' in cluster_data:
                df = cluster_data['motif_to_tf_mapping']
                table, temp_file = self.create_table_media(
                    df,
                    "Motif to Transcription Factor Database",
                    "Database mapping motif identifiers to transcription factor names"
                )
                message.append_table(table)
                temp_files.append(temp_file)
            
            # Additional analysis instructions
            analysis_instructions = """
            
            ### Analysis Instructions:
            
            **Data Integration Approach:**
            - Cross-reference accessibility patterns with known cell type markers
            - Connect TF motifs to developmental transcription factor families
            - Link associated genes to established developmental pathways
            - Consider temporal progression through zebrafish embryogenesis stages
            
            **Interpretation Framework:**
            - Use accessibility peaks as indicators of active regulatory regions
            - Interpret TF motif enrichment as evidence of regulatory control
            - Connect gene associations to functional pathway predictions
            - Ground all interpretations in zebrafish developmental biology
            
            **Output Requirements:**
            - Provide specific biological identity (not just "developmental cluster")
            - Include mechanistic hypotheses about regulatory control
            - Suggest experimental validation approaches
            - Connect to broader developmental biology concepts
            """
            
            message.append_text(analysis_instructions)
            
            # Get LLM response
            response = self.agent(message)
            
            return str(response)
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass  # Ignore cleanup errors
    
    def process_all_clusters(self, clusters_data: Dict[str, Dict], 
                           cluster_level: str = "coarse") -> None:
        """Process all clusters systematically"""
        
        self.write_to_report(f"# {cluster_level.title()} Cluster Annotations\n")
        
        total_clusters = len(clusters_data)
        print(f"Processing {total_clusters} {cluster_level} clusters...")
        
        for i, (cluster_id, cluster_data) in enumerate(clusters_data.items(), 1):
            print(f"Processing cluster {cluster_id} ({i}/{total_clusters})...")
            
            try:
                # Annotate the cluster
                annotation = self.annotate_single_cluster(cluster_id, cluster_data)
                
                # Write to report
                cluster_section = f"## Cluster {cluster_id}\n\n{annotation}\n\n"
                cluster_section += "---\n\n"  # Add separator
                self.write_to_report(cluster_section)
                
                print(f"✓ Completed cluster {cluster_id}")
                
            except Exception as e:
                error_msg = f"❌ Error processing cluster {cluster_id}: {str(e)}"
                print(error_msg)
                self.write_to_report(f"## Cluster {cluster_id}\n\n**Error**: {error_msg}\n\n")
    
    def create_meta_analysis(self, clusters_data: Dict[str, Dict]) -> None:
        """Create a comprehensive meta-analysis across all clusters"""
        
        message = Message(role="user")
        
        meta_analysis_query = f"""
        ## Comprehensive Meta-Analysis of Peak Clusters
        
        I have completed individual annotations for {len(clusters_data)} peak clusters from zebrafish 
        embryogenesis scATAC-seq data (10-24 hpf). Now I need a sophisticated meta-analysis that 
        synthesizes patterns across all clusters.
        
        **Meta-Analysis Objectives:**
        
        1. **Developmental Trajectory Reconstruction:**
           - How do clusters collectively represent the 10-24 hpf developmental progression?
           - What are the major regulatory waves and transitions?
           - Which clusters represent early vs. late developmental programs?
        
        2. **Cell Type Specification Networks:**
           - Which clusters control major cell fate decisions?
           - How do regulatory modules connect to form cell type specification cascades?
           - What are the master regulatory hierarchies?
        
        3. **Transcription Factor Regulatory Architecture:**
           - Which TF families dominate different developmental phases?
           - How do transcription factors form regulatory networks across clusters?
           - What are the key regulatory hubs and bottlenecks?
        
        4. **Biological Process Integration:**
           - How do clusters map to major developmental processes (gastrulation, neurulation, organogenesis)?
           - What are the coordinated pathway activities across development?
           - Which processes show the most regulatory complexity?
        
        5. **Comparative Developmental Biology:**
           - How do these regulatory modules compare to other vertebrate systems?
           - What are zebrafish-specific vs. conserved regulatory programs?
           - How do findings connect to broader developmental biology principles?
        
        6. **Experimental Priorities:**
           - Which clusters/TFs are highest priority for functional validation?
           - What are the most testable hypotheses from this analysis?
           - Which perturbations would have the most informative phenotypes?
        
        **Integration Requirements:**
        - Synthesize individual cluster findings into coherent developmental narrative
        - Identify cross-cluster regulatory relationships and hierarchies
        - Connect molecular patterns to developmental biology mechanisms
        - Provide framework for future experimental studies
        
        Please analyze the complete cluster annotation report and provide a comprehensive 
        meta-analysis suitable for a research paper discussion section.
        """
        
        message.append_text(meta_analysis_query)
        
        # Reference the report file for analysis
        report_reference = f"""
        ### Analysis Data Source
        Please base your meta-analysis on the individual cluster annotations contained in: {self.report_file}
        
        The report contains detailed biological interpretations for each cluster including:
        - Chromatin accessibility patterns across cell types and timepoints
        - Associated gene functions and pathway enrichments  
        - Transcription factor motif enrichments and regulatory predictions
        - Developmental biology interpretations and mechanistic hypotheses
        """
        
        message.append_text(report_reference)
        
        # Get comprehensive analysis
        response = self.agent(message)
        
        # Write meta-analysis to report
        meta_section = "# Comprehensive Meta-Analysis\n\n" + str(response)
        self.write_to_report(meta_section)


# Enhanced data loading functions
def load_cluster_data_enhanced(cluster_id: str, data_dir: str = "./cluster_data/") -> Dict:
    """
    Enhanced data loading with better error handling and data validation
    """
    cluster_data = {
        'cluster_id': cluster_id,
        'cluster_level': 'coarse',
        'parent_cluster': None,
        'n_peaks': 0
    }
    
    try:
        # Load pseudobulk accessibility data
        pseudobulk_files = [
            f"{data_dir}/cluster_{cluster_id}_pseudobulk.csv",
            f"{data_dir}/pseudobulk_cluster_{cluster_id}.csv", 
            f"{data_dir}/{cluster_id}_pseudobulk.csv"
        ]
        
        for file_path in pseudobulk_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0)
                cluster_data['pseudobulk_accessibility'] = df
                cluster_data['n_peaks'] = len(df)
                print(f"✓ Loaded pseudobulk data: {df.shape}")
                break
        
        # Load associated genes
        genes_files = [
            f"{data_dir}/cluster_{cluster_id}_genes.csv",
            f"{data_dir}/genes_cluster_{cluster_id}.csv",
            f"{data_dir}/{cluster_id}_genes.csv"
        ]
        
        for file_path in genes_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0)
                cluster_data['associated_genes'] = df
                print(f"✓ Loaded genes data: {df.shape}")
                break
        
        # Load TF motifs
        motifs_files = [
            f"{data_dir}/cluster_{cluster_id}_motifs.csv",
            f"{data_dir}/motifs_cluster_{cluster_id}.csv",
            f"{data_dir}/{cluster_id}_motifs.csv"
        ]
        
        for file_path in motifs_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0)
                cluster_data['tf_motifs'] = df
                print(f"✓ Loaded motifs data: {df.shape}")
                break
        
        # Load motif-to-TF mapping (shared across clusters)
        mapping_files = [
            f"{data_dir}/motif_tf_mapping.csv",
            f"{data_dir}/motif_to_tf_mapping.csv",
            f"{data_dir}/tf_motif_database.csv"
        ]
        
        for file_path in mapping_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0)
                cluster_data['motif_to_tf_mapping'] = df
                print(f"✓ Loaded TF mapping data: {df.shape}")
                break
                
    except Exception as e:
        print(f"❌ Error loading data for cluster {cluster_id}: {e}")
    
    return cluster_data

def process_coarse_and_fine_clusters(data_dir: str = "./cluster_data/", 
                                   use_deep_research: bool = False):
    """
    Process both coarse and fine clusters systematically
    """
    
    # Initialize the annotator
    annotator = PeakClusterAnnotator(use_deep_research=use_deep_research)
    
    # Process coarse clusters (33 clusters: C0-C32)
    print("=" * 60)
    print("PROCESSING COARSE CLUSTERS")
    print("=" * 60)
    
    coarse_cluster_ids = [f"C{i}" for i in range(33)]
    coarse_clusters_data = {}
    
    for cluster_id in coarse_cluster_ids:
        print(f"Loading data for coarse cluster {cluster_id}...")
        cluster_data = load_cluster_data_enhanced(cluster_id, data_dir)
        cluster_data['cluster_level'] = 'coarse'
        coarse_clusters_data[cluster_id] = cluster_data
    
    # Process all coarse clusters
    annotator.process_all_clusters(coarse_clusters_data, cluster_level="coarse")
    
    # Optional: Process fine clusters for specific coarse clusters
    print("=" * 60)
    print("PROCESSING FINE CLUSTERS (if available)")
    print("=" * 60)
    
    # Example: Process fine clusters for first few coarse clusters
    fine_clusters_data = {}
    for coarse_id in range(min(3, 33)):  # Process fine clusters for first 3 coarse clusters
        for fine_id in range(15):  # Up to 15 fine clusters per coarse
            fine_cluster_id = f"C{coarse_id}_F{fine_id}"
            
            # Check if fine cluster data exists
            test_file = f"{data_dir}/cluster_{fine_cluster_id}_pseudobulk.csv"
            if os.path.exists(test_file):
                print(f"Loading data for fine cluster {fine_cluster_id}...")
                cluster_data = load_cluster_data_enhanced(fine_cluster_id, data_dir)
                cluster_data['cluster_level'] = 'fine'
                cluster_data['parent_cluster'] = f"C{coarse_id}"
                fine_clusters_data[fine_cluster_id] = cluster_data
    
    if fine_clusters_data:
        annotator.process_all_clusters(fine_clusters_data, cluster_level="fine")
    
    # Create comprehensive meta-analysis
    print("=" * 60)
    print("CREATING META-ANALYSIS")
    print("=" * 60)
    
    # Combine all clusters for meta-analysis
    all_clusters_data = {**coarse_clusters_data, **fine_clusters_data}
    annotator.create_meta_analysis(all_clusters_data)
    
    print(f"✅ Complete analysis saved to: {annotator.report_file}")
    return annotator.report_file

def main():
    """Main execution function with enhanced capabilities"""
    
    # Configuration
    DATA_DIR = "./cluster_data/"  # Adjust to your data directory
    USE_DEEP_RESEARCH = True  # Set to True for more sophisticated research queries
    
    # Run the complete analysis
    report_file = process_coarse_and_fine_clusters(
        data_dir=DATA_DIR,
        use_deep_research=USE_DEEP_RESEARCH
    )
    
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Report saved to: {report_file}")
    print(f"📈 Ready for biological interpretation and experimental design!")

if __name__ == "__main__":
    main()