import os
import pandas as pd
import geopandas as gpd
import json

# should be defined in config
dict_mapDirs = {
    "zipcode": "DAGI10MULTIGEOM_SHP_20240121094449/"
    + "dagi_10m_nohist_l1.postnummerinddeling",
    "municipality": "DAGI10MULTIGEOM_SHP_20240121094449/"
    + "dagi_10m_nohist_l1.kommuneinddeling",
    "region": "DAGI10MULTIGEOM_SHP_20240121094449/"
    + "dagi_10m_nohist_l1.regionsinddeling",
}


class Geography_Helper:
    """
    Helper class used to associate zipcodes, municipalities and regions in Denmark
    To use the Class please download the following maps from kortforsyningen:
    "Danmarks Administrative Geografiske Inddeling 1:10.000"
    """

    def __init__(
        self,
    ):
        self.dir_external_data = os.path.join(os.getcwd(), "data", "external")

        self._dir_map_zipcode = os.path.join(
            self.dir_external_data, "map_zipcodes.geojson"
        )
        self._dir_map_municipality = os.path.join(
            self.dir_external_data, "map_municipality.geojson"
        )
        self._dir_map_region = os.path.join(
            self.dir_external_data, "map_region.geojson"
        )

        self._dir_dict_zipmun = os.path.join(
            self.dir_external_data, "dictionary_ZipMun.json"
        )
        self._dir_dict_munreg = os.path.join(
            self.dir_external_data, "dictionary_MunReg.json"
        )

        self._prepare_maps()
        self._create_lookup()

    def _prepare_maps(self):
        if os.path.exists(self._dir_map_zipcode):
            self.dfmap_zipcode = gpd.read_file(self._dir_map_zipcode)
            self.dfmap_municipality = gpd.read_file(self._dir_map_municipality)
            self.dfmap_region = gpd.read_file(self._dir_map_region)

        else:
            self.dfmap_zipcode = gpd.read_file(
                os.path.join(self.dir_external_data, dict_mapDirs["zipcode"])
            )
            self.dfmap_zipcode = self.dfmap_zipcode[
                ["objectid", "navn", "postnummer", "geometry"]
            ]

            self.dfmap_municipality = gpd.read_file(
                os.path.join(self.dir_external_data, dict_mapDirs["municipality"])
            )
            self.dfmap_municipality = self.dfmap_municipality[
                ["objectid", "navn", "kommunekod", "regionskod", "geometry"]
            ]

            self.dfmap_region = gpd.read_file(
                os.path.join(self.dir_external_data, dict_mapDirs["region"])
            )
            self.dfmap_region = self.dfmap_region[
                ["objectid", "navn", "regionskod", "geometry"]
            ]

            # crop zipcode polygons to only cover landmass area and not sea
            poly_dk = self.dfmap_region["geometry"].unary_union
            self.dfmap_zipcode["geometry"] = self.dfmap_zipcode[
                "geometry"
            ].intersection(poly_dk)

            self.dfmap_zipcode.to_file(self._dir_map_zipcode, driver="GeoJSON")
            self.dfmap_municipality.to_file(
                self._dir_map_municipality, driver="GeoJSON"
            )
            self.dfmap_region.to_file(self._dir_map_region, driver="GeoJSON")

    def _create_lookup(self):
        if os.path.exists(self._dir_dict_zipmun):
            with open(self._dir_dict_zipmun, "r") as fp:
                self.dict_zipmun = json.load(fp)

            with open(self._dir_dict_munreg, "r") as fp:
                self.dict_munreg = json.load(fp)

        else:
            dfmap_temp = self.dfmap_municipality.copy()
            self.dict_zipmun = {}

            for _, row_zip in self.dfmap_zipcode.iterrows():
                dfmap_temp["temp_area"] = (
                    dfmap_temp["geometry"].intersection(row_zip.geometry).area
                )
                row_mun = dfmap_temp.iloc[dfmap_temp["temp_area"].idxmax()]

                self.dict_zipmun[row_zip.postnummer] = row_mun.kommunekod

            self.dict_munreg = pd.Series(
                self.dfmap_municipality.regionskod.values,
                index=self.dfmap_municipality.kommunekod,
            ).to_dict()

            # save mappings
            with open(self._dir_dict_zipmun, "w") as fp:
                json.dump(self.dict_zipmun, fp)

            with open(self._dir_dict_munreg, "w") as fp:
                json.dump(self.dict_munreg, fp)

    def getMunicipality(self, zipcode):
        """
        Lookup best matching municipality for zipcode

        Args:
            zipcode (str): zipcode

        Return:
            str: municipality
        """
        try:
            return self.dict_zipmun[zipcode]
        except KeyError:
            return None

    def getRegion(self, municipality):
        """
        Lookup region for municipality

        Args:
            municipality (str): municipality

        Return:
            str: region
        """
        try:
            return self.dict_munreg[municipality]
        except KeyError:
            return None

    def get_dfmap(self, map_type):
        """
        Get Geopandas dataframe defining the geographical boundaries
        for the desired map_type

        Args:
            map_type (str): Type of map ['zipcode', 'municipality', 'region']

        Return
            df: Geopandas dataframe
        """
        if map_type == "zipcode":
            dfmap = self.dfmap_zipcode.copy()

        elif map_type == "municipality":
            dfmap = self.dfmap_municipality.copy()

        elif map_type == "region":
            dfmap = self.dfmap_region.copy()

        return dfmap


if __name__ == "__main__":
    a = Geography_Helper()
    df = a.get_dfmap("region")
    df = df
