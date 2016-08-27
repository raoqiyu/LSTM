import os
import pickle


def make_feature(folder, feature_type, skip_lines):
    # load features_video_appearance
    filepath = folder + '/' + feature_type
    files = os.listdir(filepath)
    files = [filepath + '/' + filename for filename in files]
    for fp in files:
        print('Processing file:', fp)
        with open(fp, 'r') as f:
            # Skip (skip_lines) lines
            # 1 @relation + 1 blank line
            #   @attribute
            # 1 @data + 1 blank line
            for i in range(skip_lines):
                f.readline()
            # [Instance_name, frameTime, n_features]
            # skip the first two element
            data = []
            for line in f:
                line = line.split(',')
                d = [float(x) for x in line[2:]]
                data.append(d)

            filename = os.path.basename(fp)
            Instance_name = 'data/' + feature_type + '/' + filename.split('_')[0] + '/' + filename.split('.')[
                0] + '.pkl'
            with open(Instance_name, 'wb') as f_saved:
                pickle.dump(data, f_saved)


def make_label(folder):
    # load labels(ratings_gold_standard)
    filepath = folder + '/ratings_gold_standard'
    labels = ['valence', 'arousal']
    for label in labels:
        labelpath = filepath + '/' + label
        files = os.listdir(labelpath)
        files = [labelpath + '/' + filename for filename in files]
        for fp in files:
            print('Processing file:', fp)
            with open(fp, 'r') as f:
                # Skip 9 lines
                # 1 @relation  + 1 blank line
                # 3 @attribute + 2 blank line
                # 1 @data  + 1 blank line
                for i in range(9):
                    f.readline()
                # [Instance_name, frameTime, GoldStandard]
                # skip the first two element
                data = []
                for line in f:
                    line = line.split(',')
                    d = float(line[-1])
                    data.append([d])

                filename = os.path.basename(fp)
                Instance_name = 'label/' + label + '/' + filename.split('_')[0] + '/' + filename.split('.')[0] + '.pkl'
                with open(Instance_name, 'wb') as f_saved:
                    pickle.dump(data, f_saved)

def make_label_text(folder):
    # load labels(ratings_gold_standard)
    filepath = folder + '/ratings_gold_standard'
    labels = ['valence', 'arousal']
    for label in labels:
        labelpath = filepath + '/' + label
        files = os.listdir(labelpath)
        files = [labelpath + '/' + filename for filename in files]
        for fp in files:
            print('Processing file:', fp)
            with open(fp, 'r') as f:
                # Skip 9 lines
                # 1 @relation  + 1 blank line
                # 3 @attribute + 2 blank line
                # 1 @data  + 1 blank line
                for i in range(9):
                    f.readline()
                # [Instance_name, frameTime, GoldStandard]
                # skip the first two element
                data = []
                for line in f:
                    line = line.split(',')
                    d = float(line[-1])
                    data.append([d])

                filename = os.path.basename(fp)
                Instance_name = 'label/txt/' + label + '/' + filename.split('_')[0] + '/' + filename.split('.')[0] + '.txt'
                with open(Instance_name, 'w') as f_saved:
                    for d in data:
                        f_saved.write(str(d[0]))
                        f_saved.write("\n")

def make_data(folder, feature_type):
    for t in feature_type:
        if t == 'features_video_geometric':
            skip_lines = 324
        else:
            skip_lines = 92
        make_feature(folder, t, skip_lines)

    make_label(folder)


def make_path(feature_type):
    # Allocate data path
    if not os.path.exists('data'):
        os.mkdir('data')
    # Allocate feature data path
    for t in feature_type:
        data_path = 'data/' + t
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        # Allocate train/dev/test data path
        for k in ['train', 'dev', 'test']:
            path = '/'.join([data_path, k])
            if not os.path.exists(path):
                os.mkdir(path)


    # Allocate label path
    if not os.path.exists('label'):
        os.mkdir('label')
    # Allocate arousal/valence path
    for label in ['arousal', 'valence']:
        path = '/'.join(['label', label])
        if not os.path.exists(path):
            os.mkdir(path)
    # Allocate train/dev/test lable path
    for label in ['arousal', 'valence']:
        for k in ['train', 'dev', 'test']:
            path = '/'.join(['label', label, k])
            if not os.path.exists(path):
                os.mkdir(path)

    # Allocate training data path
    if not os.path.exists('training'):
        os.mkdir('training')
    for t in feature_type:
        data_path = 'training/' + t
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        for label in ['arousal', 'valence', 'av']:
            path = data_path + '/' + label
            if not os.path.exists(path):
                os.mkdir(path)


def merge_data(feature_type):
    data_arousal = []
    data_valence = []
    data_av = []
    # make train data and dev data
    for k in ['train', 'dev']:
        data_path = 'data/' + feature_type + '/'
        arousal_path = 'label/arousal/'
        valence_path = 'label/valence/'
        data_path += k;
        arousal_path += k;
        valence_path += k;

        # make arousal and valence label
        filenames = os.listdir(data_path)
        filenames.sort()
        data_files = [data_path + '/' + fn for fn in filenames]
        arousal_files = [arousal_path + '/' + fn for fn in filenames]
        valence_files = [valence_path + '/' + fn for fn in filenames]

        for idx in range(len(filenames)):
            f_data, f_aro, f_val = data_files[idx], arousal_files[idx], valence_files[idx]
            with open(f_data, 'rb') as fd, open(f_aro, 'rb') as fa, open(f_val, 'rb') as fv:
                data_x = pickle.load(fd);
                data_ya = pickle.load(fa);
                data_yv = pickle.load(fv)
                data_x_ya = (data_x, data_ya)
                data_x_yv = (data_x, data_yv)
                data_x_ya_yv = (data_x, data_ya, data_yv)

                data_arousal.append(data_x_ya);
                data_valence.append(data_x_yv);
                data_av.append(data_x_ya_yv)

        save_path = 'training/' + feature_type + '/'
        with open(save_path + 'arousal/' + k + 'Arousal.pkl', 'wb') as f:
            pickle.dump(data_arousal, f)
            data_arousal = []
        with open(save_path + 'valence/' + k + 'Valence.pkl', 'wb') as f:
            pickle.dump(data_valence, f)
            data_valence = []
        with open(save_path + 'av/' + k + 'AV.pkl', 'wb') as f:
            pickle.dump(data_av, f)
            data_av = []


def load_data(path, feature, label):
    filepath = '/'.join([path, 'training', feature, label])

    f_train = open(filepath + '/' + 'train' + label.capitalize() + '.pkl', 'rb')
    trainData = pickle.load(f_train)
    f_train.close()

    f_dev = open(filepath + '/' + 'dev' + label.capitalize() + '.pkl', 'rb')
    validData = pickle.load(f_dev)
    f_dev.close()

    return trainData, validData


if __name__ == '__main__':
    AVEC = './Raw'
    feature_type = ['features_video_geometric']
   # make_path(feature_type)
    #make_data(AVEC, feature_type)
    make_label_text(AVEC)
    #merge_data('features_video_geometric')
