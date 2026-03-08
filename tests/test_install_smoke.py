import vector_engine


def test_package_import_smoke():
    assert hasattr(vector_engine, "__version__")
    assert isinstance(vector_engine.__version__, str)
    assert vector_engine.__version__
