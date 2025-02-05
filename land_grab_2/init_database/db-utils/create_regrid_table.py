import logging

from land_grab_2.init_database.db.gristdb import GristTable, GristDbField, GristDB

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

REGRID_TABLE = GristTable(name='Regrid',
                          fields=[GristDbField(name='geoid', constraints='text'),
                                  GristDbField(name='parcelnumb', constraints='text'),
                                  GristDbField(name='usecode', constraints='text'),
                                  GristDbField(name='usedesc', constraints='text'),
                                  GristDbField(name='zoning', constraints='text'),
                                  GristDbField(name='zoning_description', constraints='text'),
                                  GristDbField(name='struct', constraints='bool'),
                                  GristDbField(name='multistruct', constraints='bool'),
                                  GristDbField(name='structno', constraints='double precision'),
                                  GristDbField(name='yearbuilt', constraints='double precision'),
                                  GristDbField(name='structstyle', constraints='text'),
                                  GristDbField(name='parvaltype', constraints='text'),
                                  GristDbField(name='improvval', constraints='double precision'),
                                  GristDbField(name='landval', constraints='double precision'),
                                  GristDbField(name='parval', constraints='double precision'),
                                  GristDbField(name='agval', constraints='double precision'),
                                  GristDbField(name='saleprice', constraints='double precision'),
                                  GristDbField(name='saledate', constraints='date'),
                                  GristDbField(name='taxamt', constraints='double precision'),
                                  GristDbField(name='owntype', constraints='text'),
                                  GristDbField(name='owner', constraints='text'),
                                  GristDbField(name='ownfrst', constraints='text'),
                                  GristDbField(name='ownlast', constraints='text'),
                                  GristDbField(name='owner2', constraints='text'),
                                  GristDbField(name='owner3', constraints='text'),
                                  GristDbField(name='owner4', constraints='text'),
                                  GristDbField(name='subsurfown', constraints='text'),
                                  GristDbField(name='subowntype', constraints='text'),
                                  GristDbField(name='mailadd', constraints='text'),
                                  GristDbField(name='address_source', constraints='text'),
                                  GristDbField(name='legaldesc', constraints='text'),
                                  GristDbField(name='plat', constraints='text'),
                                  GristDbField(name='book', constraints='text'),
                                  GristDbField(name='page', constraints='text'),
                                  GristDbField(name='block', constraints='text'),
                                  GristDbField(name='lot', constraints='text'),
                                  GristDbField(name='neighborhood', constraints='text'),
                                  GristDbField(name='subdivision', constraints='text'),
                                  GristDbField(name='qoz', constraints='text'),
                                  GristDbField(name='census_block', constraints='text'),
                                  GristDbField(name='census_blockgroup', constraints='text'),
                                  GristDbField(name='census_tract', constraints='text'),
                                  GristDbField(name='sourceurl', constraints='text'),
                                  GristDbField(name='recrdareano', constraints='double precision'),
                                  GristDbField(name='gisacre', constraints='double precision'),
                                  GristDbField(name='ll_gisacre', constraints='double precision'),
                                  GristDbField(name='sqft', constraints='double precision'),
                                  GristDbField(name='ll_gissqft', constraints='double precision'),
                                  GristDbField(name='reviseddate', constraints='date'),
                                  GristDbField(name='ll_uuid', constraints='text'),
                                  GristDbField(name='padus_public_access', constraints='text'),
                                  GristDbField(name='lbcs_activity', constraints='double precision'),
                                  GristDbField(name='lbcs_activity_desc', constraints='text'),
                                  GristDbField(name='lbcs_function', constraints='double precision'),
                                  GristDbField(name='lbcs_function_desc', constraints='text'),
                                  GristDbField(name='lbcs_structure', constraints='double precision'),
                                  GristDbField(name='lbcs_structure_desc', constraints='text'),
                                  GristDbField(name='lbcs_site', constraints='double precision'),
                                  GristDbField(name='lbcs_site_desc', constraints='text'),
                                  GristDbField(name='lbcs_ownership', constraints='double precision'),
                                  GristDbField(name='lbcs_ownership_desc', constraints='text'),
                                  GristDbField(name='lat', constraints='text'),
                                  GristDbField(name='lon', constraints='text'),
                                  GristDbField(name='taxyear', constraints='text'),
                                  GristDbField(name='ll_address_count', constraints='double precision'),
                                  GristDbField(name='homestead_exemption', constraints='text'),
                                  GristDbField(name='alt_parcelnumb1', constraints='text'),
                                  GristDbField(name='alt_parcelnumb2', constraints='text'),
                                  GristDbField(name='alt_parcelnumb3', constraints='text'),
                                  GristDbField(name='parcelnumb_no_formatting', constraints='text'),
                                  GristDbField(name='plss_township', constraints='text'),
                                  GristDbField(name='plss_section', constraints='text'),
                                  GristDbField(name='plss_range', constraints='text'),
                                  GristDbField(name='geometry', constraints='json'),
                                  GristDbField(name='geometryType', constraints='text'),
                                  GristDbField(name='isRegrid', constraints='bool')])


def main():
    db = GristDB()
    log.info('Attempting to create Regrid table')
    db.create_table(REGRID_TABLE)


if __name__ == '__main__':
    main()
