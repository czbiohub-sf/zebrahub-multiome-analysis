#!/usr/bin/env python3
"""
LiteMind Peak Cluster Analysis - Main Entry Point

Run LLM-based biological interpretation of chromatin accessibility peak clusters.

Usage:
    python scripts/litemind_peak_analysis/main.py --coarse-clusters 0,1,2
    python scripts/litemind_peak_analysis/main.py --fine-clusters 0_0,0_1
    python scripts/litemind_peak_analysis/main.py --all-coarse
"""

import argparse
import sys
from pathlib import Path

from litemind import OpenAIApi, AnthropicApi
from litemind.agent.tools.builtin_tools.web_search_tool import BuiltinWebSearchTool
from litemind.agent.tools.function_tool import FunctionTool
from litemind.agent.tools.toolset import ToolSet

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.litemind_peak_analysis import config
from scripts.litemind_peak_analysis.core.coarse_cluster_analysis import CoarseClusterAnalysis
from scripts.litemind_peak_analysis.core.fine_cluster_analysis import FineClusterAnalysis
from scripts.litemind_peak_analysis.core.analysis_review import ClusterAnalysisReview
from scripts.litemind_peak_analysis.core.analysis_revision import ClusterAnalysisRevision
from scripts.litemind_peak_analysis.core.prompts import coarse_cluster_analysis_request, fine_cluster_analysis_request
from scripts.litemind_peak_analysis.bio_services.alliance import (
    fetch_alliance_expression_summary,
    fetch_alliance_gene_disease
)
from scripts.litemind_peak_analysis.bio_services.ensembl import lookup_ensembl_gene
from scripts.litemind_peak_analysis.bio_services.pubmed import fetch_pubmed_record


def setup_api(api_name=None):
    """Initialize LiteMind API based on config or argument."""
    api_to_use = api_name or config.DEFAULT_API

    if api_to_use == "openai":
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set.\n"
                "Set it with: export OPENAI_API_KEY='your-key-here'"
            )
        return OpenAIApi()
    elif api_to_use == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set.\n"
                "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
            )
        return AnthropicApi()
    else:
        raise ValueError(f"Unknown API: {api_to_use}. Must be 'openai' or 'anthropic'")


def setup_toolset():
    """Create toolset with web search and bio service tools."""
    toolset = ToolSet()

    # Add web search
    web_search = BuiltinWebSearchTool(
        max_web_searches=config.MAX_WEB_SEARCHES,
        search_context_size=config.SEARCH_CONTEXT_SIZE
    )
    toolset.add_tool(web_search)

    # Add bio service tools
    toolset.add_tool(FunctionTool(fetch_pubmed_record))
    toolset.add_tool(FunctionTool(lookup_ensembl_gene))
    toolset.add_tool(FunctionTool(fetch_alliance_expression_summary))
    toolset.add_tool(FunctionTool(fetch_alliance_gene_disease))

    return toolset


def run_coarse_cluster_analysis(cluster_id, api, toolset, output_dir, do_review=True):
    """
    Run analysis for a single coarse cluster.

    Parameters
    ----------
    cluster_id : int
        Cluster ID to analyze
    api : BaseApi
        LiteMind API instance
    toolset : ToolSet
        Tools for the agent
    output_dir : str or Path
        Output directory
    do_review : bool
        Whether to run review and revision steps

    Returns
    -------
    Task
        Final analysis task (either revision or initial analysis)
    """
    print(f"\n{'='*70}")
    print(f"Analyzing Coarse Cluster {cluster_id}")
    print(f"{'='*70}\n")

    # Run initial analysis
    analysis = CoarseClusterAnalysis(
        coarse_cluster_id=cluster_id,
        api=api,
        toolset=toolset,
        folder=output_dir
    )
    analysis()
    print(f"✓ Initial analysis complete for cluster {cluster_id}")

    # Optional: Run review and revision
    if do_review:
        print(f"  Running review for cluster {cluster_id}...")
        review = ClusterAnalysisReview(
            analysis_task=analysis,
            toolset=toolset,
            api=api,
            folder=output_dir
        )
        review()
        print(f"  ✓ Review complete")

        print(f"  Running revision for cluster {cluster_id}...")
        revision = ClusterAnalysisRevision(
            analysis_task=analysis,
            review_task=review,
            analysis_format_instructions=coarse_cluster_analysis_request,
            folder=output_dir
        )
        revision()
        print(f"  ✓ Revision complete")

        return revision

    return analysis


def run_fine_cluster_analysis(fine_cluster_id, initial_coarse_task, final_coarse_task,
                               api, toolset, output_dir, do_review=True):
    """
    Run analysis for a single fine cluster.

    Parameters
    ----------
    fine_cluster_id : str
        Fine cluster ID (e.g., "0_0")
    initial_coarse_task : Task
        Initial coarse cluster analysis task
    final_coarse_task : Task
        Final coarse cluster analysis task (after revision)
    api : BaseApi
        LiteMind API instance
    toolset : ToolSet
        Tools for the agent
    output_dir : str or Path
        Output directory
    do_review : bool
        Whether to run review and revision steps

    Returns
    -------
    Task
        Final analysis task (either revision or initial analysis)
    """
    print(f"\n{'='*70}")
    print(f"Analyzing Fine Cluster {fine_cluster_id}")
    print(f"{'='*70}\n")

    # Run initial analysis
    analysis = FineClusterAnalysis(
        fine_cluster_id=fine_cluster_id,
        initial_coarse_cluster_analysis_task=initial_coarse_task,
        final_coarse_cluster_analysis_task=final_coarse_task,
        api=api,
        toolset=toolset,
        folder=output_dir
    )
    analysis()
    print(f"✓ Initial analysis complete for fine cluster {fine_cluster_id}")

    # Optional: Run review and revision
    if do_review:
        print(f"  Running review for fine cluster {fine_cluster_id}...")
        review = ClusterAnalysisReview(
            analysis_task=analysis,
            toolset=toolset,
            api=api,
            folder=output_dir
        )
        review()
        print(f"  ✓ Review complete")

        print(f"  Running revision for fine cluster {fine_cluster_id}...")
        revision = ClusterAnalysisRevision(
            analysis_task=analysis,
            review_task=review,
            analysis_format_instructions=fine_cluster_analysis_request,
            folder=output_dir
        )
        revision()
        print(f"  ✓ Revision complete")

        return revision

    return analysis


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LiteMind-based peak cluster analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific coarse clusters
  python scripts/litemind_peak_analysis/main.py --coarse-clusters 0,1,2

  # Analyze all coarse clusters
  python scripts/litemind_peak_analysis/main.py --all-coarse

  # Analyze specific fine clusters
  python scripts/litemind_peak_analysis/main.py --fine-clusters 0_0,0_1,1_0

  # Skip review/revision steps (faster but lower quality)
  python scripts/litemind_peak_analysis/main.py --coarse-clusters 0 --no-review

  # Use Anthropic API instead of OpenAI
  python scripts/litemind_peak_analysis/main.py --coarse-clusters 0 --api anthropic
        """
    )

    parser.add_argument(
        "--coarse-clusters",
        type=str,
        help="Comma-separated list of coarse cluster IDs (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--all-coarse",
        action="store_true",
        help="Analyze all available coarse clusters"
    )
    parser.add_argument(
        "--fine-clusters",
        type=str,
        help="Comma-separated list of fine cluster IDs (e.g., '0_0,0_1,1_0')"
    )
    parser.add_argument(
        "--all-fine",
        action="store_true",
        help="Analyze all available fine clusters"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {config.RESULTS_DIR})"
    )
    parser.add_argument(
        "--api",
        type=str,
        choices=["openai", "anthropic"],
        default=None,
        help=f"LLM API to use (default: {config.DEFAULT_API})"
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip review and revision steps (faster but lower quality)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.coarse_clusters, args.all_coarse, args.fine_clusters, args.all_fine]):
        parser.error("Must specify at least one of: --coarse-clusters, --all-coarse, --fine-clusters, --all-fine")

    # Setup
    print("\n" + "="*70)
    print("LiteMind Peak Cluster Analysis")
    print("="*70)

    try:
        config.validate_config()
    except ValueError as e:
        print(f"\n{e}\n")
        sys.exit(1)

    api = setup_api(args.api)
    toolset = setup_toolset()

    output_dir = Path(args.output_dir) if args.output_dir else config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    do_review = not args.no_review

    print(f"\nConfiguration:")
    print(f"  API: {args.api or config.DEFAULT_API}")
    print(f"  Output: {output_dir}")
    print(f"  Review/Revision: {'Enabled' if do_review else 'Disabled'}")
    print(f"  Data directory: {config.DATA_DIR}")
    print()

    # Store coarse cluster results for fine cluster analysis
    coarse_results = {}

    # Run coarse cluster analyses
    if args.coarse_clusters or args.all_coarse:
        if args.all_coarse:
            cluster_ids = CoarseClusterAnalysis.get_coarse_cluster_id_list()
            print(f"Analyzing all {len(cluster_ids)} coarse clusters\n")
        else:
            cluster_ids = [int(x.strip()) for x in args.coarse_clusters.split(",")]
            print(f"Analyzing {len(cluster_ids)} coarse cluster(s): {cluster_ids}\n")

        for cluster_id in cluster_ids:
            try:
                result = run_coarse_cluster_analysis(
                    cluster_id, api, toolset, output_dir, do_review
                )
                coarse_results[cluster_id] = {
                    'initial': result if not do_review else result.analysis_task,
                    'final': result
                }
            except Exception as e:
                print(f"❌ Error analyzing coarse cluster {cluster_id}: {e}")
                continue

    # Run fine cluster analyses
    if args.fine_clusters or args.all_fine:
        if args.all_fine:
            cluster_ids = FineClusterAnalysis.get_fine_cluster_id_list()
            print(f"\nAnalyzing all {len(cluster_ids)} fine clusters\n")
        else:
            cluster_ids = [x.strip() for x in args.fine_clusters.split(",")]
            print(f"\nAnalyzing {len(cluster_ids)} fine cluster(s): {cluster_ids}\n")

        for fine_cluster_id in cluster_ids:
            try:
                # Get coarse cluster ID from fine cluster ID (e.g., "0_0" -> 0)
                coarse_id = int(fine_cluster_id.split('_')[0])

                # Check if we have the coarse cluster analysis
                if coarse_id not in coarse_results:
                    print(f"  ⚠️  Coarse cluster {coarse_id} not analyzed yet, running it now...")
                    result = run_coarse_cluster_analysis(
                        coarse_id, api, toolset, output_dir, do_review
                    )
                    coarse_results[coarse_id] = {
                        'initial': result if not do_review else result.analysis_task,
                        'final': result
                    }

                # Run fine cluster analysis
                run_fine_cluster_analysis(
                    fine_cluster_id,
                    coarse_results[coarse_id]['initial'],
                    coarse_results[coarse_id]['final'],
                    api,
                    toolset,
                    output_dir,
                    do_review
                )

            except Exception as e:
                print(f"❌ Error analyzing fine cluster {fine_cluster_id}: {e}")
                continue

    print(f"\n{'='*70}")
    print(f"✓ Analysis complete!")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
