import os


def get_filename(path: str):
    """
    './data\\FM docs 2024.3\\JOM_1998_13_4_06_The_Application_of_the_Hardin_Jones-Pauling-.pdf'
    ->
    ('./data\\FM docs 2024.3',
    'JOM_1998_13_4_06_The_Application_of_the_Hardin_Jones-Pauling-.pdf')
    """
    basepath, filename = os.path.split(path)
    return filename


def format_references(references: list[str]) -> str:
    if len(references) == 0:
        return "\nno references."
    else:
        references = [f"*{reference}*" for reference in references]
        references_str = ", ".join(references)
        return f"\nreferences: \n{references_str}"


def download_dataset():
    import openxlab
    from openxlab.dataset import get

    openxlab.login(ak=<Access Key>, sk=<Secret Key>)
    get(dataset_repo='NagatoYuki0943/FMdocs', target_path='./data/') # 数据集下载
