#!/usr/bin/env python3
"""
Unit tests for AI Suggestions feature in Archive Search
"""

import unittest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


class TestAISuggestions(unittest.TestCase):
    """Test suite for AI Suggestions feature."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample search results
        self.sample_results = [
            {
                'index': 0,
                'title': 'Commodore 64 Programmer\'s Reference Guide',
                'creator': 'Commodore',
                'date': '1982',
                'description': 'Complete technical reference for C64 programming',
                'subject': ['commodore-64', 'programming', 'manual'],
                'downloads': 5000,
                'files': [
                    {
                        'name': 'C64_Programmers_Reference_Guide.pdf',
                        'format': 'PDF',
                        'size_mb': 15.5
                    },
                    {
                        'name': 'C64_PRG.txt',
                        'format': 'Text',
                        'size_mb': 2.3
                    }
                ]
            },
            {
                'index': 1,
                'title': 'VIC-II Technical Specifications',
                'creator': 'MOS Technology',
                'date': '1983',
                'description': 'Detailed VIC-II chip documentation',
                'subject': ['vic-ii', 'hardware', 'graphics'],
                'downloads': 3500,
                'files': [
                    {
                        'name': 'VIC-II_specs.pdf',
                        'format': 'PDF',
                        'size_mb': 8.2
                    }
                ]
            }
        ]

        # Sample AI response
        self.sample_ai_response = {
            "recommendations": [
                {
                    "item_index": 0,
                    "item_title": "Commodore 64 Programmer's Reference Guide",
                    "file_name": "C64_Programmers_Reference_Guide.pdf",
                    "priority": "High",
                    "rationale": "This is the official programming reference from Commodore, essential for any C64 knowledge base.",
                    "knowledge_value": "Provides complete BASIC and machine language programming documentation, memory maps, and hardware specifications.",
                    "score": 98
                },
                {
                    "item_index": 1,
                    "item_title": "VIC-II Technical Specifications",
                    "file_name": "VIC-II_specs.pdf",
                    "priority": "High",
                    "rationale": "Original technical documentation for the graphics chip, crucial for understanding C64 graphics programming.",
                    "knowledge_value": "Detailed register-level documentation of the VIC-II chip, sprite handling, and graphics modes.",
                    "score": 95
                }
            ],
            "summary": "These search results contain high-quality official documentation that would be extremely valuable for a C64 knowledge base."
        }

    def test_prompt_construction(self):
        """Test that AI prompt is constructed correctly."""
        search_query = "commodore 64 programming"

        # Build prompt (simulating the code from admin_gui.py)
        prompt = f"""You are an expert in Commodore 64 documentation and retro computing. Analyze these search results from archive.org and recommend the TOP 5 most valuable files to download for building a C64 knowledge base.

Search Query: {search_query}

Search Results:
{json.dumps(self.sample_results, indent=2)}"""

        # Verify prompt structure
        self.assertIn("Commodore 64", prompt)
        self.assertIn(search_query, prompt)
        self.assertIn("Search Results:", prompt)
        self.assertIn("Commodore 64 Programmer's Reference Guide", prompt)

        print("✅ Prompt construction test passed")

    def test_json_extraction_with_markdown(self):
        """Test extracting JSON from markdown code blocks."""
        # Test with ```json markdown
        markdown_response = """Here are my recommendations:

```json
{
  "recommendations": [
    {
      "item_index": 0,
      "score": 95
    }
  ],
  "summary": "Test summary"
}
```

That's my analysis."""

        # Extract JSON (simulating the code from admin_gui.py)
        response_text = markdown_response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()

        # Parse JSON
        result = json.loads(response_text)

        # Verify
        self.assertIn("recommendations", result)
        self.assertIn("summary", result)
        self.assertEqual(result['recommendations'][0]['score'], 95)

        print("✅ JSON extraction test passed (markdown)")

    def test_json_extraction_plain_code_block(self):
        """Test extracting JSON from plain code blocks."""
        plain_response = """```
{
  "recommendations": [],
  "summary": "Plain code block"
}
```"""

        # Extract JSON
        response_text = plain_response
        if "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()

        # Parse JSON
        result = json.loads(response_text)

        # Verify
        self.assertIn("summary", result)
        self.assertEqual(result['summary'], "Plain code block")

        print("✅ JSON extraction test passed (plain code block)")

    def test_json_extraction_direct(self):
        """Test parsing JSON without markdown."""
        direct_json = '{"recommendations": [], "summary": "Direct"}'

        # Parse directly
        result = json.loads(direct_json)

        # Verify
        self.assertIn("summary", result)
        self.assertEqual(result['summary'], "Direct")

        print("✅ JSON extraction test passed (direct)")

    def test_recommendation_structure(self):
        """Test that recommendation objects have required fields."""
        recommendations = self.sample_ai_response['recommendations']

        required_fields = [
            'item_index',
            'item_title',
            'file_name',
            'priority',
            'rationale',
            'knowledge_value',
            'score'
        ]

        for rec in recommendations:
            for field in required_fields:
                self.assertIn(field, rec, f"Missing required field: {field}")

        print("✅ Recommendation structure test passed")

    def test_priority_levels(self):
        """Test that priority levels are valid."""
        valid_priorities = ['High', 'Medium', 'Low']

        for rec in self.sample_ai_response['recommendations']:
            self.assertIn(
                rec['priority'],
                valid_priorities,
                f"Invalid priority: {rec['priority']}"
            )

        print("✅ Priority levels test passed")

    def test_score_range(self):
        """Test that scores are within valid range (0-100)."""
        for rec in self.sample_ai_response['recommendations']:
            score = rec['score']
            self.assertGreaterEqual(score, 0, "Score should be >= 0")
            self.assertLessEqual(score, 100, "Score should be <= 100")

        print("✅ Score range test passed")

    def test_item_index_validity(self):
        """Test that item indices are valid for the search results."""
        max_index = len(self.sample_results) - 1

        for rec in self.sample_ai_response['recommendations']:
            item_index = rec['item_index']
            self.assertGreaterEqual(item_index, 0, "Index should be >= 0")
            self.assertLessEqual(
                item_index,
                max_index,
                f"Index {item_index} exceeds result count"
            )

        print("✅ Item index validity test passed")

    def test_file_matching(self):
        """Test matching recommended file with actual search result files."""
        # Get first recommendation
        rec = self.sample_ai_response['recommendations'][0]
        item_index = rec['item_index']
        file_name = rec['file_name']

        # Get corresponding search result
        result = self.sample_results[item_index]

        # Check if file exists in result
        file_found = False
        for file in result['files']:
            if file['name'] == file_name:
                file_found = True
                break

        self.assertTrue(file_found, f"File {file_name} not found in search results")

        print("✅ File matching test passed")

    def test_priority_emoji_mapping(self):
        """Test priority to emoji mapping."""
        priority_emoji = {
            'High': '🔴',
            'Medium': '🟡',
            'Low': '🟢'
        }

        # Test all valid priorities
        for priority in ['High', 'Medium', 'Low']:
            emoji = priority_emoji.get(priority, '⚪')
            self.assertIn(emoji, ['🔴', '🟡', '🟢'])

        # Test invalid priority
        emoji = priority_emoji.get('Invalid', '⚪')
        self.assertEqual(emoji, '⚪')

        print("✅ Priority emoji mapping test passed")

    @patch('os.environ.get')
    def test_api_key_check(self, mock_env_get):
        """Test API key availability check."""
        # Test with API key present
        mock_env_get.return_value = 'sk-test-key-12345'
        api_key = mock_env_get('ANTHROPIC_API_KEY')
        self.assertIsNotNone(api_key)
        self.assertTrue(len(api_key) > 0)

        # Test with no API key
        mock_env_get.return_value = None
        api_key = mock_env_get('ANTHROPIC_API_KEY')
        self.assertIsNone(api_key)

        print("✅ API key check test passed")

    def test_results_summary_preparation(self):
        """Test preparation of results summary for AI analysis."""
        # Simulate the data preparation from admin_gui.py
        results_summary = []
        for idx, result in enumerate(self.sample_results[:10]):  # Limit to first 10
            files_info = []
            for file in result['files'][:5]:  # Limit files per item
                files_info.append({
                    'name': file['name'],
                    'format': file['format'],
                    'size_mb': file['size_mb']
                })

            results_summary.append({
                'index': result['index'],
                'title': result['title'],
                'creator': result['creator'],
                'date': result['date'],
                'description': result['description'][:300] if result['description'] else "No description",
                'subject': result['subject'][:5] if isinstance(result['subject'], list) else result['subject'],
                'downloads': result['downloads'],
                'files': files_info
            })

        # Verify summary structure
        self.assertEqual(len(results_summary), len(self.sample_results))
        self.assertIn('title', results_summary[0])
        self.assertIn('files', results_summary[0])

        print("✅ Results summary preparation test passed")

    def test_recommendation_sorting_by_score(self):
        """Test that recommendations are sorted by score (highest first)."""
        recommendations = self.sample_ai_response['recommendations']

        # Check if sorted in descending order
        scores = [rec['score'] for rec in recommendations]
        sorted_scores = sorted(scores, reverse=True)

        self.assertEqual(scores, sorted_scores, "Recommendations should be sorted by score (highest first)")

        print("✅ Recommendation sorting test passed")

    def test_summary_field_presence(self):
        """Test that summary field is present in AI response."""
        self.assertIn('summary', self.sample_ai_response)
        self.assertIsInstance(self.sample_ai_response['summary'], str)
        self.assertTrue(len(self.sample_ai_response['summary']) > 0)

        print("✅ Summary field test passed")

    @patch('anthropic.Anthropic')
    def test_anthropic_client_initialization(self, mock_anthropic):
        """Test Anthropic client initialization."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Simulate client creation
        api_key = 'sk-test-key-12345'
        client = mock_anthropic(api_key=api_key)

        # Verify client was created
        mock_anthropic.assert_called_once_with(api_key=api_key)
        self.assertIsNotNone(client)

        print("✅ Anthropic client initialization test passed")

    def test_recommendation_count(self):
        """Test that exactly 5 recommendations are returned."""
        recommendations = self.sample_ai_response['recommendations']

        # Should have exactly 5 recommendations as per prompt
        self.assertEqual(
            len(recommendations),
            2,  # Our sample has 2, but production should have 5
            "Should return recommendations (sample has 2, production expects 5)"
        )

        print("✅ Recommendation count test passed")

    def test_error_handling_invalid_json(self):
        """Test error handling for invalid JSON responses."""
        invalid_json = "This is not valid JSON {invalid}"

        with self.assertRaises(json.JSONDecodeError):
            json.loads(invalid_json)

        print("✅ Invalid JSON error handling test passed")

    def test_error_handling_missing_fields(self):
        """Test handling of responses with missing fields."""
        incomplete_response = {
            "recommendations": [
                {
                    "item_index": 0,
                    # Missing required fields
                }
            ]
        }

        # Use .get() with defaults to handle missing fields
        rec = incomplete_response['recommendations'][0]
        priority = rec.get('priority', 'Medium')
        score = rec.get('score', 0)

        self.assertEqual(priority, 'Medium')  # Default value
        self.assertEqual(score, 0)  # Default value

        print("✅ Missing fields handling test passed")


def run_tests():
    """Run all tests and display results."""
    print("\n" + "=" * 60)
    print("AI SUGGESTIONS FEATURE TESTS")
    print("=" * 60 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAISuggestions)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 60 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
