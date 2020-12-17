
import pickle
import pandas as pd
import globals


def save_history(history, save_path):

    """
    Description:
        Save training model history
    Args:
        history (dict): dictionary containing training history of keras model with all captured metrics
        save_path: path to save history to
    Returns:
        None
    """

    #save history
    try:
        f = open(save_path, 'wb')
        pickle.dump(history.history, f)
        f.close()
    except pickle.UnpicklingError as e:
        print('Error', e)
    except (AttributeError,  EOFError, ImportError, IndexError) as e:
        print(traceback.format_exc(e))
    except Exception as e:
        print(traceback.format_exc(e))
        print('Error creating history pickle')


def get_model_output():

    """
    Description:
        Output model results to a CSV
    Args:
        None
    Returns:
        None
    """

    #change...
    model_output_csv = "model_output_csv_" + current_datetime +'.csv'
    model_output_csv_blob = 'models/model_output_csv_' +  current_datetime +'.csv'

    #converting model_output dictionary to pandas Dataframe
    model_output_df = pd.DataFrame(model_output, index=[0])
    #transposing model_output Dataframe
    model_output_df_t = model_output_df.transpose()
    model_output_df_t.columns = ['Values']

    #exporting Dataframe to CSV
    model_output_df_t.to_csv(model_output_csv,columns=['Values'])
