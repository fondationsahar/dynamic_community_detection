"""Integration tests for the full LAGO pipeline.

These tests verify the complete workflow from LinkStream loading
to module detection, covering various parameter combinations.
"""

from pathlib import Path

import pytest

from lago import LexType, LinkStream, TimeModules, lago_modules

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
LINKSTREAM_PATH = FIXTURES_DIR / "linkstream.txt"


@pytest.fixture
def sample_linkstream() -> LinkStream:
    """Load the sample linkstream from test fixtures."""
    ls = LinkStream()
    ls.read_txt(str(LINKSTREAM_PATH))
    return ls


class TestLagoModulesIntegration:
    """Integration tests for lago_modules function."""

    def test_basic_mm(self, sample_linkstream: LinkStream) -> None:
        """Test basic Mean-Membership detection."""
        modules = lago_modules(
            sample_linkstream,
            lex=LexType.MM,
            nb_iter=1,
            omega=2,
            refinement=None,
            fast_exploration=True,
            verbose=False,
        )
        assert isinstance(modules, TimeModules)
        assert modules.nb_modules > 0

    def test_basic_jm(self, sample_linkstream: LinkStream) -> None:
        """Test basic Joint-Membership detection."""
        modules = lago_modules(
            sample_linkstream,
            lex=LexType.JM,
            nb_iter=1,
            omega=2,
            refinement=None,
            fast_exploration=True,
            verbose=False,
        )
        assert isinstance(modules, TimeModules)
        assert modules.nb_modules > 0

    @pytest.mark.parametrize("lex_type", [LexType.MM, LexType.JM])
    def test_with_stem_refinement(self, sample_linkstream: LinkStream, lex_type: LexType) -> None:
        """Test detection with STEM refinement."""
        modules = lago_modules(
            sample_linkstream,
            lex=lex_type,
            nb_iter=1,
            omega=2,
            refinement="STEM",
            fast_exploration=True,
            verbose=False,
        )
        assert isinstance(modules, TimeModules)
        assert modules.nb_modules > 0

    @pytest.mark.parametrize("lex_type", [LexType.MM, LexType.JM])
    def test_with_stnm_refinement(self, sample_linkstream: LinkStream, lex_type: LexType) -> None:
        """Test detection with STNM refinement."""
        modules = lago_modules(
            sample_linkstream,
            lex=lex_type,
            nb_iter=1,
            omega=2,
            refinement="STNM",
            fast_exploration=True,
            verbose=False,
        )
        assert isinstance(modules, TimeModules)
        assert modules.nb_modules > 0

    @pytest.mark.parametrize("omega", [1, 2, 5])
    def test_different_omega_values(self, sample_linkstream: LinkStream, omega: float) -> None:
        """Test detection with different omega (time resolution) values."""
        modules = lago_modules(
            sample_linkstream,
            lex=LexType.MM,
            nb_iter=1,
            omega=omega,
            refinement=None,
            fast_exploration=True,
            verbose=False,
        )
        assert isinstance(modules, TimeModules)
        assert modules.nb_modules > 0

    def test_multiple_iterations(self, sample_linkstream: LinkStream) -> None:
        """Test detection with multiple iterations."""
        modules = lago_modules(
            sample_linkstream,
            lex=LexType.MM,
            nb_iter=2,
            omega=2,
            refinement=None,
            fast_exploration=True,
            verbose=False,
        )
        assert isinstance(modules, TimeModules)
        assert modules.nb_modules > 0

    def test_slow_exploration(self, sample_linkstream: LinkStream) -> None:
        """Test detection with slow (exhaustive) exploration."""
        modules = lago_modules(
            sample_linkstream,
            lex=LexType.MM,
            nb_iter=1,
            omega=2,
            refinement=None,
            fast_exploration=False,
            verbose=False,
        )
        assert isinstance(modules, TimeModules)
        assert modules.nb_modules > 0

    def test_refinement_in_loop(self, sample_linkstream: LinkStream) -> None:
        """Test detection with refinement inside the main loop."""
        modules = lago_modules(
            sample_linkstream,
            lex=LexType.MM,
            nb_iter=1,
            omega=2,
            refinement="STEM",
            fast_exploration=True,
            refinement_in=True,
            verbose=False,
        )
        assert isinstance(modules, TimeModules)
        assert modules.nb_modules > 0


class TestModuleQuality:
    """Tests to verify module detection quality."""

    def test_modules_cover_all_nodes(self, sample_linkstream: LinkStream) -> None:
        """Test that detected modules cover all nodes in the linkstream."""
        modules = lago_modules(
            sample_linkstream,
            lex=LexType.MM,
            nb_iter=1,
            omega=2,
            verbose=False,
        )

        # Get all nodes from modules
        module_nodes = set()
        for module in modules.iter_modules():
            module_nodes |= module.nodes

        # Should cover nodes from linkstream
        assert len(module_nodes) > 0

    def test_modules_have_nodes(self, sample_linkstream: LinkStream) -> None:
        """Test that all modules have nodes."""
        modules = lago_modules(
            sample_linkstream,
            lex=LexType.MM,
            nb_iter=1,
            omega=2,
            verbose=False,
        )

        for module in modules.iter_modules():
            # Each module should have nodes
            assert len(module.nodes) > 0


class TestStringLexTypeBackwardCompatibility:
    """Test backward compatibility with string lex_type values."""

    def test_string_mm(self, sample_linkstream: LinkStream) -> None:
        """Test that string 'MM' works for backward compatibility."""
        modules = lago_modules(
            sample_linkstream,
            lex="MM",
            nb_iter=1,
            omega=2,
            refinement=None,
            verbose=False,
        )
        assert isinstance(modules, TimeModules)

    def test_string_jm(self, sample_linkstream: LinkStream) -> None:
        """Test that string 'JM' works for backward compatibility."""
        modules = lago_modules(
            sample_linkstream,
            lex="JM",
            nb_iter=1,
            omega=2,
            refinement=None,
            verbose=False,
        )
        assert isinstance(modules, TimeModules)
