import pandas as pd
from pickling import pickle_get


def get_excel(df, title):
    writer = pd.ExcelWriter(f'{title}.xlsx', engine = 'xlsxwriter')
    df.to_excel(writer, title, index = False)
    
    background_color = '#0a0a23'
    font_color = 'ffffff'
    
    string_format = writer.book.add_format(
        {
            'font_color': font_color, 
            'bg_color': background_color,
            'border': 1
        }
    )
    
    dollar_format = writer.book.add_format(
        {
            'num_format': '$0.00',
            'font_color': font_color, 
            'bg_color': background_color,
            'border': 1
        }
    )
    
    integer_format = writer.book.add_format(
        {
            'num_format': '0',
            'font_color': font_color, 
            'bg_color': background_color,
            'border': 1
        }
    )
    
    column_formats = {
    'A': ['Symbol', string_format],
    'B': ['Number of Shares to Buy', integer_format],
    'C': ['Price', dollar_format]
    }
    
    for column in column_formats.keys():
        writer.sheets[title].set_column( \
                     f'{column}:{column}', 18, column_formats[column][1])
            
    writer.save()
    
"""    
df = pickle_get()
get_excel(df, 'Test')
"""
    
    