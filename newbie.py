import os
from fury import ui, window, actor
import nibabel as nib
from dipy.data import fetch_bundles_2_subjects

panel = ui.Panel2D(size=(600, 400), color=(1, 1, 1), align="right")
panel.center = (500, 400)

icon_fnames = [('square', 'model_predictions.png'), ('square2', 'wmparc.png')]

button = ui.Button2D(icon_fnames, size=(500, 300))

panel.add_element(button, coords=(0., 0.))

scene = window.Scene()

showm = window.ShowManager(scene, size=(1000, 1000))

showm.initialize()

scene.add(actor.axes())

scene.add(panel)


def change_icon_callback(i_ren, _obj, _button):
    button.next_icon()
    showm.render()


button.on_left_mouse_button_clicked = change_icon_callback


showm.start()

fetch_bundles_2_subjects()

fname_t1 = os.path.join(os.path.expanduser('~'), '.dipy',
                        'exp_bundles_and_maps', 'bundles_2_subjects',
                        'subj_1', 't1_warped.nii.gz')


img = nib.load(fname_t1)
data = img.get_data()
affine = img.affine

scene = window.Scene()
scene.background((0.5, 0.5, 0.5))
mean, std = data[data > 0].mean(), data[data > 0].std()
value_range = (mean - 0.5 * std, mean + 1.5 * std)
slice_actor = actor.slicer(data, affine, value_range)
scene.add(slice_actor)
slice_actor2 = slice_actor.copy()
slice_actor2.display(slice_actor2.shape[0]//2, None, None)
scene.add(slice_actor2)

#scene.reset_camera()
#scene.zoom(1.4)

showm = window.ShowManager(scene, size=(1000, 1000))
showm.initialize()

showm.start()


window.record(scene, out_path='slices.png', size=(600, 600),
              reset_camera=False)
