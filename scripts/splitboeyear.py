import pandas as pd
import time
import os

def main():
    data_dir = '/Users/ignasipascual/GitHub/LawGPT/raw_data/boe_year'
    filenames = sorted(os.listdir(data_dir))
    for f in filenames:
        if not f.endswith('.csv'):
            continue
        start_time = time.time()
        basename = f.split('.')[0]
        print(basename)
        os.makedirs(os.path.join(data_dir, basename), exist_ok=True)
        data = pd.read_csv(os.path.join(data_dir, f))
        for i in data.index:
            url_ = data.loc[i,'url'].split('-')
            f = f'{url_[-2]}_{int(url_[-1]):06}.csv'
            out_f = os.path.join(data_dir, basename, f)
            pd.DataFrame(data.iloc[i]).T.to_csv(out_f, index=None)
        print(f'Year completed: {int(time.time()-start_time)}')

if __name__=='__main__':
    main()