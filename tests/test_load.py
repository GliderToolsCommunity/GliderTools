from glidertools.load import voto_concat_datasets, voto_seaexplorer_nc


filename = "./tests/data/voto_nrt.nc"

# import two times to test concat
ds1 = voto_seaexplorer_nc(filename)
ds2 = voto_seaexplorer_nc(filename)


def test_dives_column_addition():
    assert len(ds1.dives) > 1


def test_voto_concat_datasets():
    ds_concat = voto_concat_datasets([ds1, ds2])
    assert 2 * len(ds1.time) == len(ds_concat.time)
