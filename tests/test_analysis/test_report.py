"""Tests for report generation (exatune.analysis.report)."""

import pytest
from pathlib import Path

from exatune.analysis.report import generate_report


class TestGenerateReport:
    """Tests for generate_report."""

    def test_creates_file_in_output_directory(self, sample_results_dataframe, temp_dir):
        """Report generation should create a file in the output directory."""
        report_path = generate_report(
            sample_results_dataframe,
            output_path=temp_dir,
            include_plots=False,
        )
        assert report_path.exists()
        assert report_path.parent == temp_dir

    def test_markdown_format_produces_md_file(self, sample_results_dataframe, temp_dir):
        """Markdown format should produce a .md file."""
        report_path = generate_report(
            sample_results_dataframe,
            output_path=temp_dir,
            format="markdown",
            include_plots=False,
        )
        assert report_path.suffix == ".md"
        content = report_path.read_text(encoding="utf-8")
        assert len(content) > 0
        assert "# ExaTune" in content

    def test_text_format_produces_txt_file(self, sample_results_dataframe, temp_dir):
        """Text format should produce a .txt file."""
        report_path = generate_report(
            sample_results_dataframe,
            output_path=temp_dir,
            format="text",
            include_plots=False,
        )
        assert report_path.suffix == ".txt"
        content = report_path.read_text(encoding="utf-8")
        assert len(content) > 0
        assert "EXATUNE LANDSCAPE ANALYSIS REPORT" in content

    def test_report_file_is_non_empty(self, sample_results_dataframe, temp_dir):
        """Generated report file should contain meaningful content."""
        report_path = generate_report(
            sample_results_dataframe,
            output_path=temp_dir,
            include_plots=False,
        )
        content = report_path.read_text(encoding="utf-8")
        assert len(content) > 100, "Report seems too short to be meaningful"
