from glidertools.load.voto_seaexplorer import voto_seaexplorer_nc


filename = "./tests/data/voto_nrt.nc"
ds = voto_seaexplorer_nc(filename)


def test_dives_column_addition():
    assert len(ds.dives) > 1
