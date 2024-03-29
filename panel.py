import numpy as np
from dipy.utils.optpkg import optional_package
import itertools
import dipy.data as dpd
import dipy.reconst.dti as dti
import matplotlib.pyplot as plt

fury, have_fury, setup_module = optional_package('fury')

if have_fury:
    from dipy.viz import actor, ui, colormap
    from dipy.viz.gmem import HORIMEM


dpd.fetch_stanford_hardi()
img, gtab = dpd.read_stanford_hardi()
data = img.get_data()


def build_label(text, font_size=18, bold=False):
    """ Simple utility function to build labels

    Parameters
    ----------
    text : str
    font_size : int
    bold : bool

    Returns
    -------
    label : TextBlock2D
    """

    label = ui.TextBlock2D()
    label.message = text
    label.font_size = font_size
    label.font_family = 'Arial'
    label.justification = 'left'
    label.bold = bold
    label.italic = False
    label.shadow = False
    label.actor.GetTextProperty().SetBackgroundColor(0, 0, 0)
    label.actor.GetTextProperty().SetBackgroundOpacity(0.0)
    label.color = (0.7, 0.7, 0.7)

    return label


def _color_slider(slider):
    slider.default_color = (1, 0.5, 0)
    slider.track.color = (0.8, 0.3, 0)
    slider.active_color = (0.9, 0.4, 0)
    slider.handle.color = (1, 0.5, 0)


def _color_dslider(slider):
    slider.default_color = (1, 0.5, 0)
    slider.track.color = (0.8, 0.3, 0)
    slider.active_color = (0.9, 0.4, 0)
    slider.handles[0].color = (1, 0.5, 0)
    slider.handles[1].color = (1, 0.5, 0)


def slicer_panel(renderer, iren,
                 data=None, affine=None,
                 world_coords=False,
                 pam=None, mask=None):
    """ Slicer panel with slicer included

    Parameters
    ----------
    renderer : Renderer
    iren : Interactor
    data : 3d ndarray
    affine : 4x4 ndarray
    world_coords : bool
        If True then the affine is applied.

    peaks : PeaksAndMetrics
        Default None

    Returns
    -------
    panel : Panel

    """

    orig_shape = data.shape
    print('Original shape', orig_shape)
    ndim = data.ndim
    tmp = data
    if ndim == 4:
        if orig_shape[-1] > 3:
            tmp = data[..., 0]
            orig_shape = orig_shape[:3]
            value_range = np.percentile(data[..., 0], q=[2, 98])
        if orig_shape[-1] == 3:
            value_range = (0, 1.)
            HORIMEM.slicer_rgb = True
    if ndim == 3:
        value_range = np.percentile(tmp, q=[2, 98])

    if not world_coords:
        affine = np.eye(4)

    image_actor_z = actor.slicer(tmp, affine=affine, value_range=value_range,
                                 interpolation='nearest', picking_tol=0.025)

    tmp_new = image_actor_z.resliced_array()

    if len(data.shape) == 4:
        if data.shape[-1] == 3:
            print('Resized to RAS shape ', tmp_new.shape)
        else:
            print('Resized to RAS shape ', tmp_new.shape + (data.shape[-1],))
    else:
        print('Resized to RAS shape ', tmp_new.shape)

    shape = tmp_new.shape

    if pam is not None:

        peaks_actor_z = actor.peak_slicer(pam.peak_dirs, None,
                                          mask=mask, affine=affine,
                                          colors=None)

    slicer_opacity = 1.
    image_actor_z.opacity(slicer_opacity)

    image_actor_x = image_actor_z.copy()
    x_midpoint = int(np.round(shape[0] / 2))
    image_actor_x.display_extent(x_midpoint,
                                 x_midpoint, 0,
                                 shape[1] - 1,
                                 0,
                                 shape[2] - 1)

    image_actor_y = image_actor_z.copy()
    y_midpoint = int(np.round(shape[1] / 2))
    image_actor_y.display_extent(0,
                                 shape[0] - 1,
                                 y_midpoint,
                                 y_midpoint,
                                 0,
                                 shape[2] - 1)

    renderer.add(image_actor_z)
    renderer.add(image_actor_x)
    renderer.add(image_actor_y)

    if pam is not None:
        renderer.add(peaks_actor_z)

    line_slider_z = ui.LineSlider2D(min_value=0,
                                    max_value=shape[2] - 1,
                                    initial_value=shape[2] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    _color_slider(line_slider_z)

    button = ui.Button2D(icon_fnames, size=(20, 30))

    def change_slice_z(slider):
        z = int(np.round(slider.value))
        HORIMEM.slicer_curr_actor_z.display_extent(0, shape[0] - 1,
                                                   0, shape[1] - 1, z, z)
        if pam is not None:
            HORIMEM.slicer_peaks_actor_z.display_extent(0, shape[0] - 1,
                                                        0, shape[1] - 1, z, z)
        HORIMEM.slicer_curr_z = z

    line_slider_x = ui.LineSlider2D(min_value=0,
                                    max_value=shape[0] - 1,
                                    initial_value=shape[0] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    _color_slider(line_slider_x)

    def change_slice_x(slider):
        x = int(np.round(slider.value))
        HORIMEM.slicer_curr_actor_x.display_extent(x, x, 0, shape[1] - 1, 0,
                                                   shape[2] - 1)
        HORIMEM.slicer_curr_x = x
        HORIMEM.window_timer_cnt += 100

    line_slider_y = ui.LineSlider2D(min_value=0,
                                    max_value=shape[1] - 1,
                                    initial_value=shape[1] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    _color_slider(line_slider_y)

    def change_slice_y(slider):
        y = int(np.round(slider.value))

        HORIMEM.slicer_curr_actor_y.display_extent(0, shape[0] - 1, y, y,
                                                   0, shape[2] - 1)
        HORIMEM.slicer_curr_y = y

    double_slider = ui.LineDoubleSlider2D(length=140,
                                          initial_values=value_range,
                                          min_value=tmp.min(),
                                          max_value=tmp.max(),
                                          shape='square')

    _color_dslider(double_slider)

    def apply_colormap(r1, r2):
        if HORIMEM.slicer_rgb:
            return

        if HORIMEM.slicer_colormap == 'disting':
            # use distinguishable colors
            rgb = colormap.distinguishable_colormap(nb_colors=256)
            rgb = np.asarray(rgb)
        else:
            # use matplotlib colormaps
            rgb = colormap.create_colormap(np.linspace(r1, r2, 256),
                                           name=HORIMEM.slicer_colormap,
                                           auto=True)
        N = rgb.shape[0]

        lut = colormap.vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(N)
        lut.SetRange(r1, r2)
        for i in range(N):
            r, g, b = rgb[i]
            lut.SetTableValue(i, r, g, b)
        lut.SetRampToLinear()
        lut.Build()

        HORIMEM.slicer_curr_actor_z.output.SetLookupTable(lut)
        HORIMEM.slicer_curr_actor_z.output.Update()

    def on_change_ds(slider):

        values = slider._values
        r1, r2 = values
        apply_colormap(r1, r2)

    double_slider.on_change = on_change_ds

    opacity_slider = ui.LineSlider2D(min_value=0.0,
                                     max_value=1.0,
                                     initial_value=slicer_opacity,
                                     length=140,
                                     text_template="{ratio:.0%}")

    _color_slider(opacity_slider)

    def change_opacity(slider):
        slicer_opacity = slider.value
        HORIMEM.slicer_curr_actor_x.opacity(slicer_opacity)
        HORIMEM.slicer_curr_actor_y.opacity(slicer_opacity)
        HORIMEM.slicer_curr_actor_z.opacity(slicer_opacity)

    volume_slider = ui.LineSlider2D(min_value=0,
                                    max_value=data.shape[-1] - 1,
                                    initial_value=0,
                                    length=140,
                                    text_template="{value:.0f}",
                                    shape='square')

    _color_slider(volume_slider)

    def change_volume(istyle, obj, slider):
        vol_idx = int(np.round(slider.value))
        HORIMEM.slicer_vol_idx = vol_idx

        renderer.rm(HORIMEM.slicer_curr_actor_x)
        renderer.rm(HORIMEM.slicer_curr_actor_y)
        renderer.rm(HORIMEM.slicer_curr_actor_z)

        tmp = data[..., vol_idx]
        image_actor_z = actor.slicer(tmp,
                                     affine=affine,
                                     value_range=value_range,
                                     interpolation='nearest',
                                     picking_tol=0.025)

        tmp_new = image_actor_z.resliced_array()
        HORIMEM.slicer_vol = tmp_new

        z = HORIMEM.slicer_curr_z
        image_actor_z.display_extent(0, shape[0] - 1,
                                     0, shape[1] - 1,
                                     z,
                                     z)

        HORIMEM.slicer_curr_actor_z = image_actor_z
        HORIMEM.slicer_curr_actor_x = image_actor_z.copy()

        if pam is not None:
            HORIMEM.slicer_peaks_actor_z = peaks_actor_z

        x = HORIMEM.slicer_curr_x
        HORIMEM.slicer_curr_actor_x.display_extent(x,
                                                   x, 0,
                                                   shape[1] - 1, 0,
                                                   shape[2] - 1)

        HORIMEM.slicer_curr_actor_y = image_actor_z.copy()
        y = HORIMEM.slicer_curr_y
        HORIMEM.slicer_curr_actor_y.display_extent(0, shape[0] - 1,
                                                   y,
                                                   y,
                                                   0, shape[2] - 1)

        HORIMEM.slicer_curr_actor_z.AddObserver('LeftButtonPressEvent',
                                                left_click_picker_callback,
                                                1.0)
        HORIMEM.slicer_curr_actor_x.AddObserver('LeftButtonPressEvent',
                                                left_click_picker_callback,
                                                1.0)
        HORIMEM.slicer_curr_actor_y.AddObserver('LeftButtonPressEvent',
                                                left_click_picker_callback,
                                                1.0)
        renderer.add(HORIMEM.slicer_curr_actor_z)
        renderer.add(HORIMEM.slicer_curr_actor_x)
        renderer.add(HORIMEM.slicer_curr_actor_y)

        if pam is not None:
            renderer.add(HORIMEM.slicer_peaks_actor_z)

        r1, r2 = double_slider._values
        apply_colormap(r1, r2)

        istyle.force_render()

    def left_click_picker_callback(obj, ev):
        ''' Get the value of the clicked voxel and show it in the panel.'''

        event_pos = iren.GetEventPosition()

        obj.picker.Pick(event_pos[0],
                        event_pos[1],
                        0,
                        renderer)

        i, j, k = obj.picker.GetPointIJK()
        res = HORIMEM.slicer_vol[i, j, k]

        # generate figure
        cc_vox = data[i, j, k]
        print(cc_vox)
        plt.plot([cc_vox])
        plt.savefig('test.png')
        icon_fnames = [('square', 'test.png'), ('square1', 'test.png')]
        # connect to button
        button = ui.Button2D(icon_fnames, size=(20, 30))
        panel.add_element(button, coords=(0., 0.))
        # make button visible

        try:
            message = '%.3f' % res
        except TypeError:
            message = '%.3f %.3f %.3f' % (res[0], res[1], res[2])
        picker_label.message = '({}, {}, {})'.format(str(i), str(j), str(k)) \
            + ' ' + message

    HORIMEM.slicer_vol_idx = 0
    HORIMEM.slicer_vol = tmp_new
    HORIMEM.slicer_curr_actor_x = image_actor_x
    HORIMEM.slicer_curr_actor_y = image_actor_y
    HORIMEM.slicer_curr_actor_z = image_actor_z

    if pam is not None:
        # change_volume.peaks_actor_z = peaks_actor_z
        HORIMEM.slicer_peaks_actor_z = peaks_actor_z

    HORIMEM.slicer_curr_actor_x.AddObserver('LeftButtonPressEvent',
                                            left_click_picker_callback,
                                            1.0)
    HORIMEM.slicer_curr_actor_y.AddObserver('LeftButtonPressEvent',
                                            left_click_picker_callback,
                                            1.0)
    HORIMEM.slicer_curr_actor_z.AddObserver('LeftButtonPressEvent',
                                            left_click_picker_callback,
                                            1.0)

    if pam is not None:
        HORIMEM.slicer_peaks_actor_z.AddObserver('LeftButtonPressEvent',
                                                 left_click_picker_callback,
                                                 1.0)

    HORIMEM.slicer_curr_x = int(np.round(shape[0] / 2))
    HORIMEM.slicer_curr_y = int(np.round(shape[1] / 2))
    HORIMEM.slicer_curr_z = int(np.round(shape[2] / 2))

    line_slider_x.on_change = change_slice_x
    line_slider_y.on_change = change_slice_y
    line_slider_z.on_change = change_slice_z

    double_slider.on_change = on_change_ds

    opacity_slider.on_change = change_opacity

    volume_slider.handle_events(volume_slider.handle.actor)
    volume_slider.on_left_mouse_button_released = change_volume

    # volume_slider.on_right_mouse_button_released = change_volume2

    line_slider_label_x = build_label(text="X Slice")
    line_slider_label_x.visibility = True
    x_counter = itertools.count()

    def label_callback_x(obj, event):
        line_slider_label_x.visibility = not line_slider_label_x.visibility
        line_slider_x.set_visibility(line_slider_label_x.visibility)
        cnt = next(x_counter)
        if line_slider_label_x.visibility and cnt > 0:
            renderer.add(HORIMEM.slicer_curr_actor_x)
        else:
            renderer.rm(HORIMEM.slicer_curr_actor_x)
        iren.Render()

    line_slider_label_x.actor.AddObserver('LeftButtonPressEvent',
                                          label_callback_x,
                                          1.0)

    line_slider_label_y = build_label(text="Y Slice")
    line_slider_label_y.visibility = True
    y_counter = itertools.count()

    def label_callback_y(obj, event):
        line_slider_label_y.visibility = not line_slider_label_y.visibility
        line_slider_y.set_visibility(line_slider_label_y.visibility)
        cnt = next(y_counter)
        if line_slider_label_y.visibility and cnt > 0:
            renderer.add(HORIMEM.slicer_curr_actor_y)
        else:
            renderer.rm(HORIMEM.slicer_curr_actor_y)
        iren.Render()

    line_slider_label_y.actor.AddObserver('LeftButtonPressEvent',
                                          label_callback_y,
                                          1.0)

    line_slider_label_z = build_label(text="Z Slice")
    line_slider_label_z.visibility = True
    z_counter = itertools.count()

    def label_callback_z(obj, event):
        line_slider_label_z.visibility = not line_slider_label_z.visibility
        line_slider_z.set_visibility(line_slider_label_z.visibility)
        cnt = next(z_counter)
        if line_slider_label_z.visibility and cnt > 0:
            renderer.add(HORIMEM.slicer_curr_actor_z)
        else:
            renderer.rm(HORIMEM.slicer_curr_actor_z)

        iren.Render()

    line_slider_label_z.actor.AddObserver('LeftButtonPressEvent',
                                          label_callback_z,
                                          1.0)

    opacity_slider_label = build_label(text="Opacity")
    volume_slider_label = build_label(text="Volume")
    picker_label = build_label(text='')
    double_slider_label = build_label(text='Colormap')

    def label_colormap_callback(obj, event):

        if HORIMEM.slicer_colormap_cnt == len(HORIMEM.slicer_colormaps) - 1:
            HORIMEM.slicer_colormap_cnt = 0
        else:
            HORIMEM.slicer_colormap_cnt += 1

        cnt = HORIMEM.slicer_colormap_cnt
        HORIMEM.slicer_colormap = HORIMEM.slicer_colormaps[cnt]
        double_slider_label.message = HORIMEM.slicer_colormap
        values = double_slider._values
        r1, r2 = values
        apply_colormap(r1, r2)
        iren.Render()

    double_slider_label.actor.AddObserver('LeftButtonPressEvent',
                                          label_colormap_callback,
                                          1.0)

    if data.ndim == 4:
        panel_size = (400, 400 + 100)
    if data.ndim == 3:
        panel_size = (400, 300 + 100)

    panel = ui.Panel2D(size=panel_size,
                       position=(850, 110),
                       color=(1, 1, 1),
                       opacity=0.1,
                       align="right")

    ys = np.linspace(0, 1, 10)

    panel.add_element(line_slider_z, coords=(0.4, ys[1]))
    panel.add_element(line_slider_y, coords=(0.4, ys[2]))
    panel.add_element(line_slider_x, coords=(0.4, ys[3]))
    panel.add_element(opacity_slider, coords=(0.4, ys[4]))
    panel.add_element(double_slider, coords=(0.4, (ys[7] + ys[8])/2.))

    if data.ndim == 4:
        if data.shape[-1] > 3:
            panel.add_element(volume_slider, coords=(0.4, ys[6]))

    panel.add_element(line_slider_label_z, coords=(0.1, ys[1]))
    panel.add_element(line_slider_label_y, coords=(0.1, ys[2]))
    panel.add_element(line_slider_label_x, coords=(0.1, ys[3]))
    panel.add_element(opacity_slider_label, coords=(0.1, ys[4]))
    panel.add_element(double_slider_label, coords=(0.1, (ys[7] + ys[8])/2.))


    if data.ndim == 4:
        if data.shape[-1] > 3:
            panel.add_element(volume_slider_label,
                              coords=(0.1, ys[6]))

    panel.add_element(picker_label, coords=(0.2, ys[5]))

    renderer.add(panel)

    # initialize colormap
    r1, r2 = value_range
    apply_colormap(r1, r2)

    return panel
