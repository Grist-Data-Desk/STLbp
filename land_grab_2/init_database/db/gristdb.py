import concurrent.futures.thread
import enum
import functools
import itertools
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import psycopg

from land_grab_2.init_database.db.db_cred import DB_CREDS

log = logging.getLogger(__name__)


@dataclass
class GristDbField:
    name: str
    constraints: str


class GristDbResults(enum.Enum):
    ONE = 'one'
    ALL = 'all'
    CALLBACK = 'callback'


class GristTable:
    def __init__(self, name: str, fields: List[GristDbField]):
        self.name = name
        self.fields = fields or []


class GristDbIndexType(enum.Enum):
    TEXT = 'text'


class GristDB:

    def execute(self,
                statement: str,
                results_type: Optional[GristDbResults] = None,
                data: Optional[Any] = None,
                callback: Optional[Any] = None):
        result = None
        try:
            with psycopg.connect(connect_timeout=60, **DB_CREDS) as conn:
                with conn.cursor() as ps_cursor:
                    if data:
                        ps_cursor.execute(statement, data)
                    else:
                        ps_cursor.execute(statement)

                    if results_type and results_type == GristDbResults.ONE:
                        result = ps_cursor.fetchone()
                        if ps_cursor.description:
                            columns = ps_cursor.description
                            result = {col.name: result[i] for i, col in enumerate(columns)}

                    if results_type and results_type == GristDbResults.ALL:
                        result = ps_cursor.fetchall()
                        if ps_cursor.description:
                            columns = ps_cursor.description
                            result = [{col.name: row[i] for i, col in enumerate(columns)} for row in result]

                    if results_type and results_type == GristDbResults.CALLBACK and callback is not None:
                        if ps_cursor.description:
                            columns = ps_cursor.description
                            for row in ps_cursor:
                                row_dict = {col.name: row[i] for i, col in enumerate(columns)}
                                callback(row_dict)

        except Exception as err:
            log.info(f'db error during txn NOT IGNORING: {err}')
            raise err

        try:
            conn.close()
        except Exception as err:
            log.info(f'failed while closing db conn with {err}')

        return result

    @staticmethod
    def _data_values_to_sql(table_fields: List[str], data: Dict[str, Any]) -> List[Any]:
        return [None if not data[k] else data[k] for k in table_fields if k in data]

    def update_table(self, table: GristTable, data: List[Dict[str, Any]]):
        if not data:
            return

        if not self.table_exists(table.name):
            raise Exception('UnknownTableError: Must create table before attempting to fill it with data.')

        raw_field_names = [f.name for f in table.fields if f.name in data[0].keys()]
        field_names = ', '.join(raw_field_names)
        row_width_sub_vars = '(' + ', '.join(['%s'] * len(raw_field_names)) + ')'
        data_len_insertion_rows = ', '.join([row_width_sub_vars] * len(data))
        insert_sql = f'INSERT INTO {table.name}({field_names}) VALUES {data_len_insertion_rows}'

        all_values = tuple(itertools.chain.from_iterable([self._data_values_to_sql(raw_field_names, d) for d in data]))

        return self.execute(insert_sql, data=all_values)

    def fetch_all_data(self, table_name: str):
        fetch_sql = f'SELECT * FROM {table_name};'
        return self.execute(fetch_sql, results_type=GristDbResults.ALL)

    def table_exists(self, table_name: str):
        exists_sql = f'''
            SELECT EXISTS ( SELECT * FROM pg_tables WHERE schemaname = 'public' AND tablename  = '{table_name}');
        '''
        return bool(self.execute(exists_sql, results_type=GristDbResults.ONE))

    def list_columns(self, table_name: str):
        list_sql = f"""
        SELECT column_name, data_type FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name   = '{table_name}';"""
        return self.execute(list_sql, results_type=GristDbResults.ALL)

    def create_table(self, table: GristTable):
        fields_sql = ',\n'.join([f"{f.name}\t{f.constraints}" for f in table.fields])
        create_table_sql = f'CREATE TABLE {table.name} (id  INT GENERATED ALWAYS AS IDENTITY, {fields_sql});'

        return self.execute(create_table_sql)

    def delete_table(self, table_name: str):
        del_table_sql = f'DROP TABLE {table_name};'
        return self.execute(del_table_sql)

    def create_index(self, table_name: str, column: str, type: Optional[GristDbIndexType] = None):
        index_sql = f'CREATE INDEX {column}_idx ON {table_name} ({column});'
        if type == GristDbIndexType.TEXT:
            index_sql = f"CREATE INDEX {column}_gin_idx ON {table_name} USING GIN (to_tsvector('english', {column}));"
        return self.execute(index_sql)

    def delete_index(self, column: str, type: Optional[GristDbIndexType] = None):
        index_sql = f'DROP INDEX {column}_idx;'
        if type == GristDbIndexType.TEXT:
            index_sql = f'DROP INDEX {column}_gin_idx;'
        return self.execute(index_sql)

    def list_indexes(self):
        list_sql = f'''
        SELECT
            tablename,
            indexname,
            indexdef
        FROM
            pg_indexes
        WHERE
            schemaname = 'public'
        ORDER BY
            tablename,
            indexname;
        '''
        return self.execute(list_sql, results_type=GristDbResults.ALL)

    def copy_to(self, table: GristTable, data: List[Dict[str, Any]]):
        raw_field_names = [f.name for f in table.fields if f.name in data[0].keys()]
        rows = [tuple([d[f] for f in raw_field_names]) for d in data]

        field_names = ', '.join(raw_field_names)

        try:
            with psycopg.connect(connect_timeout=60, **DB_CREDS) as conn:
                with conn.cursor() as ps_cursor:
                    with ps_cursor.copy(f"COPY {table.name} ({field_names}) FROM STDIN;") as copy:
                        [copy.write_row(row) for row in rows]

        except Exception as err:
            log.info(f'db error during write NOT IGNORING: {err}')
            raise err

        try:
            conn.close()
        except Exception as err:
            log.info(f'failed while closing db conn with {err}')

    def search_column_value_in_set(self,
                                   table_name: str,
                                   column_name: str,
                                   search_items: List[str],
                                   callback: Optional[Any] = None):
        formatted_search_items = search_items
        row_width_sub_vars = ', '.join([f"'{i}'" for i in formatted_search_items])

        search_sql = f"""
        SELECT * FROM {table_name} WHERE {column_name} IN ({row_width_sub_vars});
        """
        if callback:
            return self.execute(search_sql, results_type=GristDbResults.CALLBACK, callback=callback)

        return self.execute(search_sql, results_type=GristDbResults.ALL)

    def fetch_from_col_where_val(self,
                                 select_col,
                                 where_col,
                                 val,
                                 distinct: bool = False,
                                 pagination_row_id: Optional[str] = None,
                                 limit: int = None):
        distinct_clause = 'DISTINCT' if distinct else 'id,'
        order_by_col = select_col if distinct else 'id'
        limit_clause = '' if not limit else f'LIMIT {limit}'
        list_all_sql = f"SELECT {distinct_clause} {select_col} FROM regrid WHERE {where_col} = '{val}' ORDER BY {order_by_col} ASC {limit_clause};"
        if pagination_row_id:
            list_all_sql = f"""
            SELECT {distinct_clause} {select_col} FROM regrid WHERE {where_col} = '{val}' AND id > {pagination_row_id} ORDER BY {order_by_col} ASC {limit_clause};
            """

        return self.execute(list_all_sql, results_type=GristDbResults.ALL)

    def count_where(self,
                    where_col,
                    val):
        list_all_sql = f"SELECT COUNT(id) FROM regrid WHERE {where_col} = '{val}';"
        return self.execute(list_all_sql, results_type=GristDbResults.ONE)

    @functools.lru_cache
    def ids_where(self, where_col, val, batch_size=None):
        limit_constraint = '' if not batch_size else f'LIMIT {batch_size}'
        list_all_sql = f"SELECT id FROM regrid WHERE {where_col} = '{val}' ORDER BY id ASC {limit_constraint};"
        results = self.execute(list_all_sql, results_type=GristDbResults.ALL)
        all_results = results
        if batch_size:
            while results:
                page_id = results[-1]['id']
                list_all_sql_paged = f"""
                SELECT id FROM regrid WHERE id > {page_id} AND {where_col} = '{val}' ORDER BY id ASC LIMIT {batch_size};
                """
                results = self.execute(list_all_sql_paged, results_type=GristDbResults.ALL)
                all_results += results
        return all_results

    def hydrate_ids(self, search_items):
        row_width_sub_vars = ', '.join([f"'{i}'" for i in search_items])
        list_all_sql = f"SELECT * FROM regrid WHERE id IN ({row_width_sub_vars});"
        return self.execute(list_all_sql, results_type=GristDbResults.ALL)

    def fetch_all_unique(self, select_col):
        list_all_sql = f"SELECT DISTINCT {select_col} FROM regrid  ORDER BY {select_col} ASC;"
        return self.execute(list_all_sql, results_type=GristDbResults.ALL)

    def crs_search_by_state(self, state):
        list_all_sql = f'''SELECT * 
                        FROM epsg_coordinatereferencesystem 
                        WHERE coord_ref_sys_name ILIKE '%{state}%';'''
        return self.execute(list_all_sql, results_type=GristDbResults.ALL)

    def search_text_col_has_query_paged_impl(self,
                                             table_name: str,
                                             column_name: str,
                                             queries: List[str],
                                             exclusion_ids: Optional[List[str]] = None,
                                             pagination_row_id: Optional[str] = None,
                                             limit: int = None):
        if not queries:
            return None
        limit_clause = '' if not limit else f'LIMIT {limit}'
        exclusion_ids = exclusion_ids or []

        proper_quotes_queries = (q.replace("'", "''") for q in queries)
        query_template = f"to_tsvector('english', {column_name}) @@ "
        predicates = ' OR '.join([
            (query_template + f"plainto_tsquery('english', '{q}')").format((column_name, q))
            for q in proper_quotes_queries
        ])

        exclusion_ids_fmttd = ', '.join([f"'{i}'" for i in exclusion_ids])
        id_exclude_predicate = '' if not exclusion_ids else f'AND id NOT IN ({exclusion_ids_fmttd})'

        search_sql = f"""
            SELECT * 
            FROM {table_name} 
            WHERE {predicates}
            {id_exclude_predicate}
            ORDER BY id ASC {limit_clause};
        """
        if pagination_row_id:
            search_sql = f"""
                SELECT * 
                FROM {table_name} 
                WHERE {predicates}
                {id_exclude_predicate}
                AND id > {pagination_row_id} ORDER BY id ASC {limit_clause};
            """

        return self.execute(search_sql, results_type=GristDbResults.ALL)

    def search_text_column_has_query(self,
                                     table_name: str,
                                     column_name: str,
                                     queries: List[str],
                                     exclusion_ids: Optional[List[str]] = None,
                                     pagination_row_id: Optional[str] = None,
                                     limit: int = 500):
        result = self.search_text_col_has_query_paged_impl(table_name,
                                                           column_name,
                                                           queries,
                                                           exclusion_ids,
                                                           pagination_row_id,
                                                           limit)
        all_results = result
        while result:
            pagination_row_id = result[-1]['id']
            result = self.search_text_col_has_query_paged_impl(table_name,
                                                               column_name,
                                                               queries,
                                                               exclusion_ids,
                                                               pagination_row_id,
                                                               limit)
            all_results += result
        return all_results

    def _db_state_by_min_col_length(self,
                                    state: str,
                                    column: str,
                                    min_len: int,
                                    max_len: int,
                                    pagination_row_id: Optional[str] = None,
                                    limit: int = 10000) -> List[Dict[str, Any]]:

        # TODO add query contains filter
        search_sql = f"""
        select * 
        from regrid 
        where state2 = '{state}' 
        AND length({column}) >= {min_len} 
        AND length({column}) <= {max_len}
        ORDER BY id ASC 
        LIMIT {limit};
        """

        if pagination_row_id:
            search_sql = f"""
                    select * 
                    from regrid 
                    where state2 = '{state}' 
                    AND length({column}) >= {min_len} 
                    AND length({column}) <= {max_len} 
                    AND id > {pagination_row_id} 
                    ORDER BY id ASC 
                    LIMIT {limit};
                    """

        return self.execute(search_sql, results_type=GristDbResults.ALL)

    def state_by_min_col_length(self, state: str, column: str, min_len: int, max_len: int) -> List[Dict[str, Any]]:
        result = self._db_state_by_min_col_length(state=state, column=column, min_len=min_len, max_len=max_len)
        all_results = result
        while result:
            pagination_row_id = result[-1]['id']
            result = self._db_state_by_min_col_length(state=state,
                                                      column=column,
                                                      min_len=min_len,
                                                      max_len=max_len,
                                                      pagination_row_id=pagination_row_id)
            all_results += result

        return all_results

    def db_query_field_in_value_by_ids_1(self, queries, column, id_batch):
        predicates = ' OR '.join([f"{column} ILIKE '%{q}%'" for q in queries])
        id_batch = [record['id'] for record in id_batch]
        inclusion_ids_fmttd = ', '.join([str(i) for i in id_batch])

        ids_sql = f"SELECT DISTINCT id  FROM regrid WHERE id IN ({inclusion_ids_fmttd}) AND ({predicates});"
        results_ids = self.execute(ids_sql, results_type=GristDbResults.ALL)
        return results_ids

    def db_query_field_in_value_by_ids_2(self, results_ids):
        if not results_ids:
            return results_ids

        results_ids = [record['id'] for record in results_ids]
        results_ids_fmttd = ', '.join([str(i) for i in results_ids])

        search_sql = f"""SELECT * FROM regrid WHERE id IN  ({results_ids_fmttd});"""
        results_details = self.execute(search_sql, results_type=GristDbResults.ALL)

        return results_details
