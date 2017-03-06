"""
make audio and video data into one data for modality-fusion training

"""

import  pickle
em = ['arousal', 'valence']
#feature = ['features_audio', 'features_video']
#part = ['train', 'valid', 'test']
data_base = '.'
feature = ['features_audio', 'features_video_appearance']
parts = ['train', 'dev']
#part = ['train', 'valid', 'test']
for e in em:
    for part in parts:
        audio_file = '/'.join([data_base,'training', feature[0], e, part])+e.capitalize()+'AfterSecond.pkl'
        video_file = '/'.join([data_base,'training',  feature[1], e, part])+e.capitalize()+'AfterSecond.pkl'
        print('\n',audio_file,'\n',video_file)
        audio_fp = open(audio_file,'rb')
        video_fp = open(video_file,'rb')

        audio_data = pickle.load(audio_fp)
        video_data = pickle.load(video_fp)

        n_audioSamples, tmp, n_audioSteps, n_audioFeatureSize = len(audio_data), len(audio_data[0]), len(audio_data[0][0]), \
                                                                len(audio_data[0][0][0])

        n_videoSamples, tmp, n_videoSteps, n_videoFeatureSize = len(audio_data), len(audio_data[0]), len(audio_data[0][0]), \
                                                                len(audio_data[0][0][0])

        print(n_audioSamples, tmp, n_audioSteps, n_audioFeatureSize)
        print(n_videoSamples, tmp, n_videoSteps, n_videoFeatureSize)

        if [n_audioSamples, tmp, n_audioSteps, n_audioFeatureSize] != [n_videoSamples, tmp, n_videoSteps, n_videoFeatureSize]:
            print('data wrong')
            break
        n_samples, _, n_steps, n_featureSize =  n_videoSamples, tmp, n_videoSteps, n_videoFeatureSize

        fusion_data = []
        linear_data = []
        linear_label = []
        for i in range(n_samples):
            if audio_data[i][1] != video_data[i][1]:
                print('label wrong')
                break
            ad = audio_data[i][0]
            vd = video_data[i][0]
            label = audio_data[i][1]
            print('(',len(ad),len(ad[0]),')', end=" ")
            print('(',len(vd),len(vd[0]),')', end=" ")
            print('(',len(label),len(label[0]),')')
            fd = []
            for j in range(n_audioSteps):
                fd.append([ad[j][0], vd[j][0]])
                linear_data.append([ad[j][0], vd[j][0]])
                linear_label.append((label[j][0]))
            fusion_data.append([fd,label])

        print(len(fusion_data), len(fusion_data[0]), len(fusion_data[0][0]),len(fusion_data[0][0][0]),len(fusion_data[0][1][0]))
        file_name = data_base+'/training/fusion2/'+e+'/'+part+e.capitalize()+'.pkl'
        print('Save as ', file_name)
        fp = open(file_name, 'wb')
        pickle.dump(fusion_data, fp)
        fp.close()

        file_name = data_base+'/training/fusion2/'+e+'/'+part+e.capitalize()+'linear.pkl'
        print('Save as ', file_name)
        fp = open(file_name, 'wb')
        pickle.dump(linear_data, fp)
        pickle.dump(linear_label, fp)
        fp.close()