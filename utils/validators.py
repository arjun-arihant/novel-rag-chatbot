"""
Validators for checking system requirements and novel format
"""
import os
import re
import subprocess
from typing import Dict, List, Tuple


class Validators:
    """Validate system setup and novel format."""

    @staticmethod
    def check_ollama_running() -> Tuple[bool, str]:
        """
        Check if Ollama is running.

        Returns:
            (success, message) tuple
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                return True, "Ollama is running"
            else:
                return False, "Ollama is not responding properly"
        except FileNotFoundError:
            return False, "Ollama is not installed. Please install from https://ollama.com"
        except subprocess.TimeoutExpired:
            return False, "Ollama is not responding (timeout)"
        except Exception as e:
            return False, f"Error checking Ollama: {str(e)}"

    @staticmethod
    def check_models_downloaded(models: List[str]) -> Tuple[bool, str]:
        """
        Check if required Ollama models are downloaded.

        Args:
            models: List of model names to check

        Returns:
            (success, message) tuple
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return False, "Could not list Ollama models"

            output = result.stdout.lower()
            missing_models = []

            for model in models:
                # Handle model tags (e.g., "mistral:7b" -> check for "mistral")
                model_base = model.split(':')[0].lower()
                if model_base not in output:
                    missing_models.append(model)

            if missing_models:
                models_str = ", ".join(missing_models)
                return False, f"Missing models: {models_str}. Run 'ollama pull <model>' for each."
            else:
                return True, f"All required models are available: {', '.join(models)}"

        except Exception as e:
            return False, f"Error checking models: {str(e)}"

    @staticmethod
    def validate_novel_format(novel_path: str) -> Tuple[bool, str, Dict]:
        """
        Validate novel file format and structure.

        Args:
            novel_path: Path to novel file

        Returns:
            (success, message, details) tuple
        """
        if not os.path.exists(novel_path):
            return False, f"Novel file not found: {novel_path}", {}

        try:
            with open(novel_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                return False, "Novel file is empty", {}

            # Check for chapter headers
            chapter_pattern = re.compile(r'Chapter\s+(\d+):\s+(.+?)\n', re.IGNORECASE)
            matches = list(chapter_pattern.finditer(content))

            if not matches:
                return False, "No chapters found. Chapters must follow format: 'Chapter N: Title'", {}

            # Validate chapter numbering
            chapter_numbers = [int(m.group(1)) for m in matches]

            details = {
                "total_chapters": len(chapter_numbers),
                "total_characters": len(content),
                "first_chapter": chapter_numbers[0] if chapter_numbers else None,
                "last_chapter": chapter_numbers[-1] if chapter_numbers else None,
                "chapter_numbers": chapter_numbers
            }

            # Check for gaps or duplicates
            issues = []

            # Check for duplicates
            if len(chapter_numbers) != len(set(chapter_numbers)):
                duplicates = [num for num in set(chapter_numbers) if chapter_numbers.count(num) > 1]
                issues.append(f"Duplicate chapter numbers: {duplicates}")

            # Check for large gaps (more than 1)
            sorted_nums = sorted(chapter_numbers)
            for i in range(len(sorted_nums) - 1):
                if sorted_nums[i + 1] - sorted_nums[i] > 1:
                    issues.append(f"Gap in chapter numbering: Chapter {sorted_nums[i]} to {sorted_nums[i + 1]}")

            if issues:
                return True, f"Novel format valid but has issues: {'; '.join(issues)}", details
            else:
                return True, f"Novel format is valid. Found {len(matches)} chapters.", details

        except UnicodeDecodeError:
            return False, "Novel file encoding error. Please ensure file is UTF-8 encoded.", {}
        except Exception as e:
            return False, f"Error validating novel: {str(e)}", {}

    @staticmethod
    def check_directory_structure(base_path: str = ".") -> Tuple[bool, str]:
        """
        Check if required directories exist or can be created.

        Args:
            base_path: Base directory path

        Returns:
            (success, message) tuple
        """
        required_dirs = ["chroma_db", "chroma_db_backup"]
        issues = []

        for dir_name in required_dirs:
            dir_path = os.path.join(base_path, dir_name)

            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)
                except Exception as e:
                    issues.append(f"Cannot create {dir_name}: {str(e)}")

        if issues:
            return False, "; ".join(issues)
        else:
            return True, "Directory structure is valid"

    @staticmethod
    def check_file_permissions(file_paths: List[str]) -> Tuple[bool, str]:
        """
        Check if required files are readable/writable.

        Args:
            file_paths: List of file paths to check

        Returns:
            (success, message) tuple
        """
        issues = []

        for path in file_paths:
            if os.path.exists(path):
                if not os.access(path, os.R_OK):
                    issues.append(f"Cannot read {path}")
                if not os.access(path, os.W_OK):
                    issues.append(f"Cannot write to {path}")

        if issues:
            return False, "; ".join(issues)
        else:
            return True, "All file permissions are correct"

    @staticmethod
    def validate_config(config: Dict) -> Tuple[bool, str]:
        """
        Validate configuration dictionary.

        Args:
            config: Configuration dict

        Returns:
            (success, message) tuple
        """
        required_keys = [
            ("paths", "novel"),
            ("models", "embedding"),
            ("models", "llm"),
            ("chunking", "size"),
            ("retrieval", "top_k")
        ]

        missing = []
        for keys in required_keys:
            value = config
            path = []
            for key in keys:
                path.append(key)
                if not isinstance(value, dict) or key not in value:
                    missing.append(".".join(path))
                    break
                value = value[key]

        if missing:
            return False, f"Missing required config keys: {', '.join(missing)}"

        # Validate value ranges
        issues = []

        chunk_size = config.get("chunking", {}).get("size", 0)
        if chunk_size < 100 or chunk_size > 10000:
            issues.append("chunk_size should be between 100 and 10000")

        top_k = config.get("retrieval", {}).get("top_k", 0)
        if top_k < 1 or top_k > 50:
            issues.append("top_k should be between 1 and 50")

        if issues:
            return False, f"Invalid config values: {'; '.join(issues)}"

        return True, "Configuration is valid"

    @staticmethod
    def run_all_checks(config: Dict) -> Dict[str, Tuple[bool, str]]:
        """
        Run all validation checks.

        Args:
            config: Configuration dict

        Returns:
            Dict mapping check names to (success, message) tuples
        """
        results = {}

        # Check Ollama
        results["ollama_running"] = Validators.check_ollama_running()

        # Check models
        models = [
            config.get("models", {}).get("embedding", ""),
            config.get("models", {}).get("llm", "")
        ]
        models = [m for m in models if m]  # Filter empty strings
        results["models_downloaded"] = Validators.check_models_downloaded(models)

        # Check novel format
        novel_path = config.get("paths", {}).get("novel", "novel.txt")
        success, message, details = Validators.validate_novel_format(novel_path)
        results["novel_format"] = (success, message)

        # Check directory structure
        results["directory_structure"] = Validators.check_directory_structure()

        # Check config
        results["config_valid"] = Validators.validate_config(config)

        return results

    @staticmethod
    def print_validation_report(results: Dict[str, Tuple[bool, str]]):
        """
        Print a formatted validation report.

        Args:
            results: Results from run_all_checks
        """
        print("\n" + "=" * 60)
        print("SYSTEM VALIDATION REPORT")
        print("=" * 60 + "\n")

        all_passed = True

        for check_name, (success, message) in results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status} | {check_name.replace('_', ' ').title()}")
            print(f"       {message}")
            print()

            if not success:
                all_passed = False

        print("=" * 60)

        if all_passed:
            print("✓ All checks passed! System is ready.")
        else:
            print("✗ Some checks failed. Please address the issues above.")

        print("=" * 60 + "\n")

        return all_passed
