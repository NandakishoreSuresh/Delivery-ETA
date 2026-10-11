from src.features.feature_config import (
    FEATURES,
    POST_OUTCOME_COLUMNS,
    IDENTIFIER_COLUMNS,
    TEMPORARILY_EXCLUDED_COLUMNS,
)


def test_no_post_outcome_feature_is_used():
    assert not set(FEATURES) & set(POST_OUTCOME_COLUMNS)


def test_no_identifier_feature_is_used():
    assert not set(FEATURES) & set(IDENTIFIER_COLUMNS)


def test_prep_time_is_not_currently_used():
    assert "restaurant_preparation_time_minutes" not in FEATURES
    assert "restaurant_preparation_time_minutes" in TEMPORARILY_EXCLUDED_COLUMNS
