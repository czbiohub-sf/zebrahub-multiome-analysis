import json
import requests
import pandas as pd

class FishEnrichR:
    def __init__(self, base_url='http://maayanlab.cloud/FishEnrichr'):
        self.base_url = base_url
        
    def analyze_genes(self, genes, description="Gene list"):
        """Submit gene list for analysis"""
        genes_str = '\n'.join(genes)
        payload = {
            'list': (None, genes_str),
            'description': (None, description)
        }
        response = requests.post(f'{self.base_url}/addList', files=payload)
        if not response.ok:
            raise Exception('Error analyzing gene list')
        return json.loads(response.text)
    
    def view_gene_list(self, user_list_id):
        """View submitted gene list"""
        response = requests.get(f'{self.base_url}/view?userListId={user_list_id}')
        if not response.ok:
            raise Exception('Error getting gene list')
        return json.loads(response.text)
    
    def get_enrichment(self, user_list_id, gene_set_library):
        """Get enrichment analysis results"""
        response = requests.get(
            f'{self.base_url}/enrich?userListId={user_list_id}&backgroundType={gene_set_library}'
        )
        if not response.ok:
            raise Exception('Error fetching enrichment results')
        return json.loads(response.text)
    
    def download_results(self, user_list_id, filename, gene_set_library):
        """Download enrichment results to file"""
        url = f'{self.base_url}/export?userListId={user_list_id}&filename={filename}&backgroundType={gene_set_library}'
        response = requests.get(url, stream=True)
        if not response.ok:
            raise Exception('Error downloading results')
            
        with open(f'{filename}.txt', 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)