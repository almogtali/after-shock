import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os
import re
from collections import Counter

def extract_questions_in_column_a(file_path, question_list, fuzzy_rate=0.8):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Initialize a dictionary to store questions for each sheet
    extracted_questions = {}
    
    # Iterate through each sheet
    for sheet_name in xls.sheet_names:
        # Read the sheet
        df = xls.parse(sheet_name, header=None)
        
        # Focus only on column A (first column)
        column_a = df.iloc[:, 0]
        
        # Store matched questions
        matched_questions = []
        
        for index, cell_value in column_a.items():
            if pd.notnull(cell_value):  # Ensure the cell is not empty
                # Perform fuzzy matching
                match, score = process.extractOne(
                    str(cell_value), question_list, scorer=fuzz.ratio
                )
                if score / 100 >= fuzzy_rate:
                    matched_questions.append((index, match))
        
        if matched_questions:
            extracted_questions[sheet_name] = matched_questions
    
    return extracted_questions
def calculate_fuzzy_rate(input_string, list_string):
    results = [(item, fuzz.ratio(input_string, item) / 100) for item in list_string]
    for res in results:
        if res[1]>0.85:
            return (True,res[0])
    return (False,None)

def find_next_null(df,index1,min_rows=5):
    for i in range(index1 + min_rows, len(df)):
        if pd.notnull(df.iloc[i, 3]):
            return i

def find_table_border(data, start_row, start_col=0, max_allowed_none=1):
    n_rows = len(data)
    current_row = start_row
    consecutive_none_count = 0

    # Check from the starting row until the end of the table
    while current_row < n_rows:
        row_values = data.iloc[current_row, start_col:start_col+1]
        if row_values.isnull().all():
            # Count consecutive empty rows
            consecutive_none_count += 1
            if consecutive_none_count > max_allowed_none:
                break  # Stop if too many consecutive `None` rows
        else:
            consecutive_none_count = 0  # Reset the count if the row is not fully empty
        current_row += 1

    # Adjust end_row (exclude the consecutive None rows from the table)
    end_row = current_row - consecutive_none_count
    return start_row, end_row

def make_path_to_valied(path):
    return path.replace('.','').replace(',','').replace('?','').replace('=','')


def make_to_precent(count:Counter)->list:
    total = sum(count.values())
    result = {}
    for key, count in count.items():
        result[key] = count / total
    return result

def parse_xlsx(file_path,path_q,column=0):
    # Load the Excel file
    xls_q = pd.ExcelFile(path_q)
    for sheet_name in xls_q.sheet_names:
        question_list =xls_q.parse(sheet_name, header=0)['question'].to_list()
    xls = pd.ExcelFile(file_path)
    # Initialize a dictionary to store results
    extracted_tables = {}
    # Iterate through each sheet in the workbook
    collect = []
    for sheet_name in xls.sheet_names:
        # Read the sheet
        try:
            df = xls.parse(sheet_name, header=None)
            
            # Get merged cells information
            wb = xls.book
            sheet = wb[sheet_name]
            df = pd.DataFrame(sheet.values)
            saver = {}
            for index,row in df.iterrows():
                #[row,column]
                ceil =df.loc[index,column]
                if pd.isna(ceil):continue
                try:
                    res = calculate_fuzzy_rate(ceil,question_list)
                    if res[0]:
                        for i in range(3,0,-1):
                            start_row, end_row = find_table_border(df,index+1,start_col=1,max_allowed_none=i)
                            if end_row-start_row < 40:break

                        df_temp = df.loc[start_row-1:end_row,:]
                        # start_col, end_col = find_table_border(df_temp.T,0,start_col=0,max_allowed_none=2)
                        # df_temp = df_temp.loc[:,start_col:end_col]
                        df_to_save = df_temp.T
                        range_ = df_to_save.columns
                        
                        df_to_save.columns = range_-range_.start
                        collect.append(df_to_save)
                        # if res[1] in saver.keys():
                        #     saver[res[1]].append(df_temp)
                        # else:
                        #     saver[res[1]] = [df_temp]
                        # base = 'Results\\'+make_path_to_valied(res[1])+"\\"
                        # save_dir = os.path.dirname(base)
                        # if save_dir and not os.path.exists(save_dir):
                        #     os.makedirs(save_dir)
                        # save_path = save_dir+'\\'+f"_{sheet_name}"+file_path.split('\\')[1]
                        # while os.path.exists(save_path):
                        #     save_path = save_path.replace('.xlsx','_more.xlsx')
                        # df_temp.to_excel(save_path)

                        # find_next_null(df,start_row,min_rows=5)
                except Exception as e:
                    pass
                    tables = []
        except Exception as e:
            pass
    pd.concat(collect,ignore_index=True).to_excel('temp.xlsx')
        

def find_xlsx_files(directory='data'):
    xlsx_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx'):
                xlsx_files.append(os.path.join(root, file))
    return xlsx_files

def values_as_values(dict_):
    res = {}
    for val in dict_.keys():
        res[val]=val
    return res

def make_pivots(path):
    xls = pd.ExcelFile(path)
    collect = []
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        category_column = df.columns[:6]
        qustions = df.columns[6:]
        # wb = xls.book
        # sheet = wb[sheet_name]
        # df = pd.DataFrame(sheet.values)
        pivot_tables = {}
        resutlts = []
        for col in qustions:
            first = True
            for cat in category_column:
                for cat_value in df[cat].unique():

                    c = Counter(df[df[cat] == cat_value][col])
                    if col not in pivot_tables.keys() and False:
                        resutlts  = [{'cat': cat,'subcat':cat_value,'values':make_to_precent(c)}]
                        pivot_tables[col]  = [{'q':col,'cat': cat,'subcat':cat_value,'values':make_to_precent(c)}]
                    # else:
                    if first:
                        first=False
                        resutlts.append({'q':col,'cat': cat,'subcat':cat_value,**values_as_values(make_to_precent(c))})
                        resutlts.append({'q':col,'cat': cat,'subcat':cat_value,**make_to_precent(c)})
                    else:
                        resutlts.append({'q':col,'cat': cat,'subcat':cat_value,**make_to_precent(c)})
                        # pivot_tables[col].append({'cat': cat,'subcat':cat_value,'values':make_to_precent(c)})
    pd.DataFrame(resutlts).to_excel('df.xlsx')
    return

def combain_almog(directory_path='almog'):
    excel_data = []
    xls = pd.ExcelFile('almog.xlsx')
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name,index_col=0)
    for filename in os.listdir(directory_path):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_excel(file_path,index_col=0)
            # df['date'] = filename.split('.')[0]
            # df.to_excel(file_path)
            excel_data.append(df)

def combain_all_version(directory_path='clean'):
    sheets_names = []
    excel_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_excel(file_path)
            if df.columns[0]!='q_full':
                df = pd.read_excel(file_path,index_col=0)
            excel_data.append(df)
            # xls = pd.ExcelFile(file_path)
            # if len(xls.sheet_names)>1:
            #     print(filename)
            #     print(xls.sheet_names)
            #     print('-'*120)
            #     sheets_names.extend(xls.sheet_names)
    # print(set(sheets_names))
    pd.concat(excel_data,ignore_index=True).to_excel('temps.xlsx')

def fix_questions():
    df = pd.read_excel('aldates.xlsx',index_col=0)
    # df_q = pd.read_excel('fix_questions.xlsx')
    # translate =  df_q.set_index('q_full')['change_with'].to_dict()
    # df['q_full'] = df['q_full'].map(lambda x:translate[x] if x in translate.keys() else x)
    df['q_full'] = df['q_full'].apply(lambda x:x.strip() if isinstance(x,str) else x)
    df.to_excel('amon.xlsx')
    df['q_full'].value_counts().to_excel('fix_questions.xlsx')

def fix_subject():
    df = pd.read_excel('temps.xlsx',index_col=0)
    df_q = pd.read_excel('sub.xlsx')
    translate =  df_q.set_index('subject')['change_with'].to_dict()
    df['subject'] = df['subject'].map(lambda x:translate[x] if x in translate.keys() else x)
    df.to_excel('temps.xlsx')
    df['subject'].value_counts().to_excel('sub.xlsx')
    pass

def fix_sub_subject():
    df = pd.read_excel('temps.xlsx',index_col=0)
    df_q = pd.read_excel('sub.xlsx')
    translate =  df_q.set_index('sub_subject')['change_with'].to_dict()
    df['sub_subject'] = df['sub_subject'].map(lambda x:translate[x] if x in translate.keys() else x)
    df.to_excel('temps.xlsx')
    df['sub_subject'].value_counts().to_excel('sub.xlsx')
    pass

def filter_questions():
    df_all = pd.read_excel('main.xlsx',index_col=0)
    question = pd.read_excel('q.xlsx')
    filtered_df = df_all[df_all['q_full'].isin(question['question'].to_list())]
    # for q in question['question'].to_list():
    #     filted = df_all[df_all['q_full'] == q]
    #     filted.reset_index().to_excel(f'{q}.xlsx')

    filtered_df.to_excel('solid_q.xlsx')

if __name__=="__main__":
    import sys
    # fix_questions()
    # combain_all_version()
    filter_questions()
    sys.exit(0)
    # combain_almog()
    question_path = "q.xlsx"
    all_paths = find_xlsx_files()
    check_only = ['24_09']
    for path in all_paths:
        do = False
        for check in check_only:
            if check in path or check=='ALL':
                do=True
                break
        if do:
            # for i in range(3):
            # pivots = make_pivots(path)
            parse_xlsx(path, question_path,column=0)

    # Display extracted tables
    # for sheet, data in tables.items():
    #     print(f"Sheet: {sheet}")
    #     for entry in data:
    #         print(f"Question: {entry['question']}")
    #         print(f"Table:\n{entry['table']}")
