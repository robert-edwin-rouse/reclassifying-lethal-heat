"""
Configuration for the data paths and random forest model to reproduce the
results in the Reclassifying Lethal Heat paper along with a list of features
and formatted plotting labels.
"""

from dataclasses import dataclass


'''
Your email address and api urls for data collection.
'''
email = 'youremail@somedomain.com'
era5_url = 'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2'


'''
Unprocessed and preprocessed data file paths.
'''
lhw_data = 'data/lethal_heatwave_input.csv'
unp_data = 'data/UN_population_pyramid_data.csv'
bmi_data = 'data/Lancet_BMI_data.csv'
ghx_data = 'data/GHX_SDI_data.csv'

lethal_heat_data = 'data/lethal_heatwaves.csv'


'''
Specification of the model parameters, plotting parameters, and input variables
with labels for plotting.
'''
@dataclass
class ForestConfig:
    class_weights = {0:1, 1:3}
    tree_depth: int = 50
    n_trees: int = 200
    train_split: float = 0.8
    val_split: float = 0.1
    random_seed: int = 42
    n_neighbors: int = 5
    platt_threshold: int = 0.6
    platt_method: str = 'sigmoid'
    date_column: str = 'StartDate'
    colour_1A: str = 'indianred'
    colour_1B: str = 'darkred'
    colour_2A: str = 'cadetblue'
    colour_2B: str = 'darkslategray'

target = ['Documented_Mortality']
feature_dict = {'Max_Temperature':'Max Temp',
                'Mean_Humidity':'Mean Humid',
                'Mean_Windspeed':'Mean Windspeed',
                'Delta_Temp_30':'∆Temp'+r'$_{30}$',
                'Delta_Temp_90':'∆Temp'+r'$_{90}$',
                'Delta_Temp_180':'∆Temp'+r'$_{180}$',
                'Adaptive_Temperature':'Adaptive Temp',
                'Delta_Humid_30':'∆Humid'+r'$_{30}$',
                'Delta_Humid_90':'∆Humid'+r'$_{90}$',
                'Delta_Humid_180':'∆Humid'+r'$_{180}$',
                'Adaptive_Humidity':'Adaptive Humid',
                'Mean_Age':'Mean Age',
                'Age_Gradient':'Age Gradient',
                'Mean_BMI':'Mean BMI',
                'Mean_SDI':'Mean SDI'}


'''
Output paths for generated datasets and figures.
'''
grid_filepath = 'data/forest_grid_search.csv'
results_filepath = 'data/main_results.csv'
feature_builder_filepath = 'data/feature_builder.csv'
ablation_filepath = 'data/ablation.csv'
permutation_filepath = 'data/permutation.csv'

model_filepath = 'assets/HeatwaveClassifier.joblib'
model_dictpath = 'assets/HeatwaveClassifierNormalisation.joblib'

fig1_path = 'assets/Figure1.eps'
fig2a_path = 'assets/Figure2a.eps'
fig2b_path = 'assets/Figure2b.eps'
fig2c_path = 'assets/Figure2c.eps'
fig2d_path = 'assets/Figure2d.eps'
fig2e_path = 'assets/Figure2e.eps'
fig2f_path = 'assets/Figure2f.eps'
fig3a_path = 'assets/Figure3a.eps'
fig3b_path = 'assets/Figure3b.eps'
fig3c_path = 'assets/Figure3c.eps'
fig3d_path = 'assets/Figure3d.eps'
fig3e_path = 'assets/Figure3e.eps'
fig3f_path = 'assets/Figure3f.eps'
fig4a_path = 'assets/Figure4a.eps'
fig4b_path = 'assets/Figure4b.eps'
fig5_path = 'assets/Figure5.eps'
fig6a_path = 'assets/Figure6a.eps'
fig6b_path = 'assets/Figure6b.eps'
fig6c_path = 'assets/Figure6c.eps'
fig6d_path = 'assets/Figure6d.eps'
fig7a_path = 'assets/Figure7a.eps'
fig7b_path = 'assets/Figure7b.eps'
fig7c_path = 'assets/Figure7c.eps'
fig7d_path = 'assets/Figure7d.eps'
fig8_path = 'assets/Figure8.eps'
figa1a_path = 'assets/FigureA1a.eps'
figa2_path = 'assets/FigureA2.eps'
figa3_path = 'assets/FigureA3.eps'
figa4a_path = 'assets/FigureA4a.eps'
figa4b_path = 'assets/FigureA4b.eps'