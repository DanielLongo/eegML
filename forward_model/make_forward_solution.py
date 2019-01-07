import mne
# x = np.ones((10,4))
# z = mne.io.RawArray(x, info=mne.create_info(10, 1))
# mne.make_forward_solution(z, None, None, None)

#From sample data

raw = mne.io.read_raw_fif("/Users/DanielLongo/mne_data/MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif")
trans = None
# src = mne.setup_source_space("sample", subjects_dir="/Users/DanielLongo/mne_data/MNE-sample-data/subjects", spacing="oct6")
sphere = (0, 0, 0, 120)
subj = 'sample'
subjects_dir = '/Users/DanielLongo/mne_data/MNE-sample-data/subjects'
aseg_fname = subjects_dir + '/sample/mri/aseg.mgz'
volume_label = 'Left-Cerebellum-Cortex'
src = mne.setup_volume_source_space(subj, mri=aseg_fname, sphere=sphere,
	volume_label=volume_label,
	subjects_dir=subjects_dir)
bem = "/Users/DanielLongo/mne_data/MNE-sample-data/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif"
fwd = mne.make_forward_solution(raw.info, trans, src, bem)


#bert
to make bem file model = mne.make_bem_model('bert', subjects_dir="./bert") 
write_bem_surfaces('sample-5120-5120-5120-bem.fif', model) from http://martinos.org/mne/stable/manual/cookbook.html#flow-diagram
# raw is from guassine distribtuion shape X
# trans = None WHAT SHOULD THIS BE SET TO?
# src = mne.setup_source_space("bert", subjects_dir="../bert")
# bem =  "../bert/bem/ A BEM FILE
