import xlrd
from openpyxl import Workbook

def convert_xls_to_xlsx(input_file, output_file):
    # 打开旧的 .xls 文件
    xls_book = xlrd.open_workbook(input_file)
    xls_sheet = xls_book.sheet_by_index(2)  # 指定转换哪个工作表？

    # 创建一个新的 .xlsx 文件
    wb = Workbook()
    ws = wb.active

    # 复制数据
    for row in range(xls_sheet.nrows):
        for col in range(xls_sheet.ncols):
            ws.cell(row=row + 1, column=col + 1, value=xls_sheet.cell_value(row, col))

    # 保存为 .xlsx 文件
    wb.save(output_file)

# 文件路径
input_file ='D:/competition/2024数学建模国赛/2011B/cumcm2011B附件2_全市六区交通网路和平台设置的数据表.xls'
output_file = 'D:/competition/2024数学建模国赛/2011B/jianmoshuju3.xlsx'

# 执行转换
convert_xls_to_xlsx(input_file, output_file)
