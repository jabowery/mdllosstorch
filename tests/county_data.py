import os
import regex as re
import pandas as pd
import numpy as np

# Population variable sequence exception
POPULATION_SEQUENCE = {
    'conceptual_variable': 'total_population',
    'year_ranges': {
        'PIN030': (1969, 1970),
        'PST015': (1971, 1979),
        'PST025': (1981, 1989), 
        'PST035': (1990, 1999),
        'PST045': (2000, 2009),
        'AGE010': (2010, 2010)
    }
}

def get_year_from_column(col):
    """Extract year from column name using the specified format"""
    yrcode = col[-4:-1]  # Get 3-digit year code
    year_str = yrcode[0] + ('9' if yrcode[0] == '1' else '0') + yrcode[-2:]
    return int(year_str)

def get_population_variable_for_year(year):
    """Get the appropriate population variable for a given year"""
    for var_id, (start_year, end_year) in POPULATION_SEQUENCE['year_ranges'].items():
        if start_year <= year <= end_year:
            return var_id
    return None

def get_df_numeric_clean():
    # Read the main data CSV file
    path = (
        os.getenv("LOTC_CENSUS_BZ2")
    )
    if path:
        print('decompresing bz2')
        df = pd.read_csv(path, sep="\t", compression='bz2', index_col='STCOU').drop(0, errors='ignore') #and drop national row
    else:
        df = pd.read_csv('lotc.csv', sep="\t", index_col='STCOU').drop(0, errors='ignore') #and drop national row

    df.columns.name = "Ecological Variable"

    # Convert all columns to numeric, coercing errors to NaN
    df_numeric = df.apply(pd.to_numeric, errors='coerce')

    # Drop original columns with any NaN values (before transformations)
    df_numeric_clean = df_numeric.dropna(axis=1).astype(np.float64)
    return df_numeric_clean

# Get all population columns available based on first 6 characters and year code
def is_population_column(col):
    """Check if column is a population variable for its year"""
    try:
        first_6 = col[:6]
        year = get_year_from_column(col)
        pop_var = get_population_variable_for_year(year)
        return pop_var is not None and first_6 == pop_var
    except (ValueError, IndexError):
        return False
    
def getdf():
    df_numeric_clean = get_df_numeric_clean()
    df = df_numeric_clean

    # Read the metadata file to get unit types for each column
    try:
        # Read the metadata CSV file to get column unit types
        dir_path = os.path.dirname(os.path.realpath(__file__))
        metadata_df = pd.read_csv(dir_path+'/Ref/MastdataExtended.csv', sep='\t')
        
        # Create a dictionary mapping Item_Id to Unit_Indicator
        unit_mapping = dict(zip(metadata_df['Item_Id'], metadata_df['Unit_Indicator']))
        desc_mapping = dict(zip(metadata_df['Item_Id'], metadata_df['Item_Description']))
        
        # Define which unit types should NOT be divided by population
        no_divide_units = ['RTE', 'PCT', 'SYM', 'RNK']
        prenormalized_descriptions = r'average|avg|mean|median'
        
        # Remove negative values that interfere with normalization
        df_numeric_clean = df_numeric_clean.loc[:, ~(df < 0).any()]
        
        available_pop_cols = [col for col in df_numeric_clean.columns if is_population_column(col)]
        
        if available_pop_cols:
            drop_columns = []
            
            # For each column, determine the appropriate population variable to use
            for col in df_numeric_clean.columns:
                try:
                    # Extract year from column name
                    year = get_year_from_column(col)
                    pop_var = get_population_variable_for_year(year)
                    
                    # Get the unit type for this column
                    unit_type = unit_mapping.get(col, None)
                    
                    # Skip if this is a population column itself
                    if is_population_column(col):
                        continue
                        
                    # Check if we should normalize this column
                    if (unit_type is not None and 
                        unit_type not in no_divide_units and 
                        pop_var is not None and
                        not re.findall(prenormalized_descriptions, desc_mapping.get(col, ''))):
                        
                        # Find the appropriate population column for this year
                        pop_col = None
                        for potential_pop_col in df_numeric_clean.columns:
                            if (potential_pop_col[:6] == pop_var and 
                                get_year_from_column(potential_pop_col) == year):
                                pop_col = potential_pop_col
                                break
                        
                        if pop_col is not None:
                            # Normalize by the appropriate population variable
                            population = df_numeric_clean[pop_col]
                            df_numeric_clean.loc[:, col] = (df_numeric_clean[col] / population).astype(np.float64)
                        
                    elif unit_type != 'PCT':
                        # Drop columns that are not normalizable
                        drop_columns.append(col)
                    else:
                        # Keep PCT columns as-is
                        df_numeric_clean.loc[:, col] = df_numeric_clean[col]
                        
                except (ValueError, IndexError):
                    # If we can't extract year, decide based on unit type only
                    unit_type = unit_mapping.get(col, None)
                    if unit_type != 'PCT' and unit_type not in no_divide_units:
                        drop_columns.append(col)
            
            # Remove population columns from drop list and add them back
            drop_columns = [col for col in drop_columns if not is_population_column(col)]
            df_numeric_clean = df_numeric_clean.drop(drop_columns, axis=1)
            
        else:
            print("Warning: No population columns found. No per-capita normalization applied.")
    
    except Exception as e:
        print(f"Warning: Could not apply per-capita normalization due to error: {e}")
    
    df = df_numeric_clean
    df = df.loc[:, ~df.apply(np.isinf).any()]
    df = df.dropna(axis=0)
    df = df.loc[:, ~(df < 0).any()]
    df = df.dropna(axis=0).astype(np.float64) # drop states that apparently had 0 population(?)
    
    return df

def getpaneldf_transforms():
    # 1) grab the raw panel and its numpy values
    base_df   = getpaneldf()                  # cols: (var, STCOU)
    base_vals = base_df.values                # shape (T, P)

    # 2) compute all transforms in one go
    sqrt_vals = np.sqrt(base_vals)            # ident, sqrt, log1p
    log_vals  = np.log1p(base_vals)

    # 3) stack horizontally: [ident | sqrt | log]
    all_vals = np.hstack([base_vals, sqrt_vals, log_vals])  # shape (T, 3P)

    # 4) rebuild a MultiIndex with levels [var, transform, STCOU]
    vars_   = base_df.columns.get_level_values(0)
    stcus_  = base_df.columns.get_level_values(1)
    P       = len(vars_)

    # repeat each block
    var_all       = np.tile(vars_, 3)
    stcou_all     = np.tile(stcus_, 3)
    transform_all = ['ident']*P + ['sqrt']*P + ['log']*P

    mi = pd.MultiIndex.from_arrays(
        [var_all, transform_all, stcou_all],
        names=['var', 'transform', 'STCOU']
    )

    # 5) put it all back into a DataFrame
    return pd.DataFrame(all_vals, index=base_df.index, columns=mi)


def getdf_transforms():
    # Use all the variables for transformations
    base_vars_df = getdf()

    # Prepare lists for constructing the new DataFrame with hierarchical columns
    all_new_columns_data = []
    all_new_column_names = []

    for var_name in base_vars_df.columns:
        original_series = base_vars_df[var_name]

        # 'ident' transformation (original values)
        all_new_columns_data.append(original_series)
        all_new_column_names.append((var_name, 'ident'))

        # 'sqrt' transformation
        # np.sqrt of negative numbers results in NaN.
        sqrt_transformed_series = np.sqrt(original_series)
        all_new_columns_data.append(sqrt_transformed_series)
        all_new_column_names.append((var_name, 'sqrt'))

        # 'log' transformation
        # np.log of negative numbers results in NaN. np.log(0) results in -np.inf.
        log_transformed_series = np.log(original_series)
        # Replace inf/-inf (e.g., from log(0)) with NaN for consistent dropping logic
        log_transformed_series = log_transformed_series.replace([np.inf, -np.inf], np.nan)
        all_new_columns_data.append(log_transformed_series)
        all_new_column_names.append((var_name, 'log'))

    # Create the new DataFrame with hierarchical columns
    hierarchical_df = pd.concat(all_new_columns_data, axis=1)
    hierarchical_df.columns = pd.MultiIndex.from_tuples(all_new_column_names, names=['Ecological Variable', 'Transformation'])

    # Drop specific transformed sub-columns that contain any NaN values
    columns_to_drop = []
    for col_tuple in hierarchical_df.columns:
        if hierarchical_df[col_tuple].isnull().any():
            columns_to_drop.append(col_tuple)
    
    hierarchical_df_cleaned = hierarchical_df.drop(columns=columns_to_drop)
    
    # Ensure the row index name 'STCOU' is preserved
    hierarchical_df_cleaned.index.name = 'STCOU'
    df = hierarchical_df_cleaned[-hierarchical_df_cleaned.isna().any(axis=1)] 
    df = df[-np.isinf(df).any(axis=1)]

    return df

def getpaneldf(filepath='lotc_panel.feather'):
    import numpy as np, pandas as pd
    from scipy.signal import savgol_filter
    print('reading df')
    #filled_df = pd.read_csv('lotc_filled.csv', sep='\t')
    import pandas as pd, numpy as np
    import pandas as pd
    print('read')
    df = pd.read_feather('lotc_panel.feather')

    print('index STCOU')
    # --- 2.1  put STCOU back on the index ------------------------------
    df = df.set_index('STCOU')                       # now rows = counties
    print('multiindex ')
    # --- 2.2  rebuild the column MultiIndex ----------------------------
    def col2varyearint(col):
        var, yearstr = col.split('__')
        return var,int(yearstr)
    df.columns = pd.MultiIndex.from_tuples(
        (col2varyearint(col) for col in df.columns),     # ('AFN110','1969'), …
        names=['var', 'year']
    )

    # 1 ─ swap the column levels so that 'year' is outermost, then sort
    step1 = (
        df.swaplevel(0, 1, axis=1)      # columns become (year , var)
          .sort_index(axis=1, level=0)  # keep years in ascending order
    )

    # 2 ─ move 'year' from columns to the row index, then spread STCOU into columns
    panel = (
        step1.stack('year', future_stack=True)             # index (STCOU , year) ; cols = var
              .unstack('STCOU')         # index  year           ; cols = (var , STCOU)
              .sort_index()             # 1930 … 2010
              .astype('float32', copy=False)   # keep it lean
    )
    return panel

def putpaneldf(panel_df, filepath):
    from pathlib import Path
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True)
    vars_  = panel_df.columns.levels[0]     # 1 411 variables
    stcou_ = panel_df.columns.levels[1]     # counties
    years_ = panel_df.index

    full_cols = pd.MultiIndex.from_product(
                   [vars_, years_],     # years vary fastest → matches reshape order
                   names=['var', 'year']
                )
    panel_df = panel_df.stack(level=0, future_stack=True).transpose()
    # --- 1.1  flatten the MultiIndex columns --------------------------
    panel_df.columns = [f'{yr}__{var}' for var, yr in panel_df.columns]

    # --- 1.2  move the row index to a *properly named* column ----------
    panel_df = panel_df.reset_index(names='STCOU')     # not "year"!
    # --- 1.3  write to Feather (no index stored) -----------------------
    panel_df.to_feather(filepath, compression='zstd')          # fast & compact


class TemporalCountyData:
    """
    A class that provides temporal access to county data with proper population sequence handling.
    >>> from county_data import TemporalCountyData
    >>> temporal_df = TemporalCountyData()
    >>> temporal_df
    TemporalCountyData with 3137 counties and 4683 variable-year combinations
    Variable_ID    AFN110                         AFN120  ... total_population
    Year             1997      2002      2007       1997  ...             2006       2007       2008       2009
    STCOU                                                 ...
    1000         0.001592  0.001582  0.001745   0.888700  ...        4597688.0  4637904.0  4677464.0  4708708.0
    1001         0.001212  0.001484  0.001846   0.677773  ...          49105.0    49834.0    50354.0    50756.0
    1003         0.002328  0.002153  0.002396   1.723441  ...         168516.0   172815.0   176212.0   179878.0
    1005         0.001478  0.001502  0.001513   0.607214  ...          29556.0    29736.0    29836.0    29737.0
    1007         0.000619  0.000810  0.000884   0.000000  ...          21285.0    21485.0    21589.0    21587.0
    ...               ...       ...       ...        ...  ...              ...        ...        ...        ...
    56037        0.002815  0.002715  0.002747   1.581956  ...          37948.0    39320.0    39942.0    41226.0
    56039        0.011123  0.009345  0.008967  10.274873  ...          19666.0    20073.0    20541.0    20710.0
    56041        0.002508  0.002658  0.002541   1.216403  ...          19631.0    20071.0    20537.0    20927.0
    56043        0.004103  0.004309  0.004228   0.980424  ...           7671.0     7805.0     7807.0     7911.0
    56045        0.003569  0.004283  0.004237   0.616952  ...           6568.0     6845.0     6928.0     7009.0

    [3137 rows x 4683 columns]
    >>> temporal_df._original_df
    Ecological Variable  AFN110197D  AFN110202D  AFN110207D  ...  WTN270197D  WTN270202D  WTN270207D
    STCOU                                                    ...                                    
    1000                   0.001592    0.001582    0.001745  ...    0.650931    0.682828    0.957492
    1001                   0.001212    0.001484    0.001846  ...    0.000000    0.000000    0.000000
    1003                   0.002328    0.002153    0.002396  ...    0.242486    0.482999    0.000000
    1005                   0.001478    0.001502    0.001513  ...    0.000000    0.000000    0.000000
    1007                   0.000619    0.000810    0.000884  ...    0.000000    0.053853    0.000000
    ...                         ...         ...         ...  ...         ...         ...         ...
    56037                  0.002815    0.002715    0.002747  ...    0.410530    0.475550    0.941404
    56039                  0.011123    0.009345    0.008967  ...    0.000000    0.184707    0.000000
    56041                  0.002508    0.002658    0.002541  ...    0.176624    0.000000    0.645409
    56043                  0.004103    0.004309    0.004228  ...    0.339819    0.000000    0.121076
    56045                  0.003569    0.004283    0.004237  ...    0.067212    0.076629    0.205844

    [3137 rows x 4683 columns]
    >>> temporal_df.temporal2originalcol
    {('AFN110', 1997): 'AFN110197D', ('AFN110', 2002): 'AFN110202D', ('AFN110', 2007): 'AFN110207D',...}
    >>> temporal_df.temporal2originalcol
    """
    
    def __init__(self):
        self._original_df = getdf()
        self.temporal2originalcol = {} # to get back to the original column names
        self._temporal_df = self._create_temporal_multiindex_df()
    
    def _create_temporal_multiindex_df(self):
        """Create a MultiIndex DataFrame with (Variable_ID, Year) columns"""
        # Start with the original dataframe
        df = self._original_df.copy()
        
        # Create list to store tuples for MultiIndex
        temporal_columns = []
        temporal_data = []
        
        # Process each column in the original dataframe
        for col in df.columns:
            try:
                # Extract year from column name
                year = get_year_from_column(col)
                
                # Check if this is a population variable that should be renamed
                if is_population_column(col):
                    # Use 'total_population' as the conceptual variable name
                    temporal_columns.append(('total_population', year))
                else:
                    # Keep year independent original variable name
                    temporal_columns.append((col[:-4], year))
                
                temporal_data.append(df[col])
                
            except (ValueError, IndexError):
                # If we can't extract year, keep as-is but assign a default year
                temporal_columns.append((col, None))
                temporal_data.append(df[col])
            self.temporal2originalcol[temporal_columns[-1]] = col
        
        # Create the temporal DataFrame with MultiIndex columns
        temporal_df = pd.concat(temporal_data, axis=1)
        temporal_df.columns = pd.MultiIndex.from_tuples(
            temporal_columns, 
            names=['Variable_ID', 'Year']
        )
        
        # Sort by Variable_ID and Year for better organization
        temporal_df = temporal_df.sort_index(axis=1)
        
        return temporal_df
    
    def __getattr__(self, name):
        """Delegate DataFrame methods to temporal_df by default"""
        return getattr(self._temporal_df, name)
    
    def __getitem__(self, key):
        """Enable df[key] syntax on temporal structure"""
        return self._temporal_df[key]
    
    def __len__(self):
        """Return length of temporal_df"""
        return len(self._temporal_df)
    
    def __repr__(self):
        """String representation showing temporal structure"""
        return f"TemporalCountyData with {len(self._temporal_df)} counties and {len(self._temporal_df.columns)} variable-year combinations\n{self._temporal_df.__repr__()}"
    
    def __str__(self):
        """String representation"""
        return self.__repr__()
    
    @property
    def loc(self):
        """Enable df.loc syntax on temporal structure"""
        return self._temporal_df.loc
    
    @property
    def iloc(self):
        """Enable df.iloc syntax on temporal structure"""
        return self._temporal_df.iloc
    
    @property
    def index(self):
        """Access to temporal_df index"""
        return self._temporal_df.index
    
    @property
    def columns(self):
        """Access to temporal_df columns"""
        return self._temporal_df.columns
    
    @property
    def values(self):
        """Access to temporal_df values"""
        return self._temporal_df.values
    
    @property
    def shape(self):
        """Access to temporal_df shape"""
        return self._temporal_df.shape
    
    def as_atemporal(self):
        """Return original format DataFrame"""
        return self._original_df.copy()
    
    def get_variable_years(self, variable_id):
        """Get all years available for a specific variable"""
        if variable_id in self._temporal_df.columns.get_level_values(0):
            return self._temporal_df[variable_id].columns.tolist()
        else:
            return []
    
    def get_year_variables(self, year):
        """Get all variables available for a specific year"""
        year_mask = self._temporal_df.columns.get_level_values(1) == year
        return self._temporal_df.columns[year_mask].get_level_values(0).tolist()
    
    def get_population_for_year(self, year):
        """Get population data for a specific year"""
        if ('total_population', year) in self._temporal_df.columns:
            return self._temporal_df[('total_population', year)]
        else:
            # Try to find the appropriate population variable for this year
            pop_var = get_population_variable_for_year(year)
            if pop_var and (pop_var, year) in self._temporal_df.columns:
                return self._temporal_df[(pop_var, year)]
            return None
    
    def summary(self):
        """Provide a summary of the temporal data structure"""
        variables = self._temporal_df.columns.get_level_values(0).unique()
        years = self._temporal_df.columns.get_level_values(1).unique()
        years = [y for y in years if y is not None]  # Remove None values
        
        print(f"TemporalCountyData Summary:")
        print(f"Counties: {len(self._temporal_df)}")
        print(f"Variables: {len(variables)}")
        print(f"Years: {min(years) if years else 'N/A'} - {max(years) if years else 'N/A'}")
        print(f"Total variable-year combinations: {len(self._temporal_df.columns)}")
        
        # Show population sequence coverage
        print(f"\nPopulation sequence coverage:")
        for var_id, (start_year, end_year) in POPULATION_SEQUENCE['year_ranges'].items():
            available_years = []
            for year in range(start_year, end_year + 1):
                if ('total_population', year) in self._temporal_df.columns:
                    available_years.append(year)
            print(f"  {var_id} ({start_year}-{end_year}): {len(available_years)} years available")
