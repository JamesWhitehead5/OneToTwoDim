from src.propagated_basis_optimization import *


def run_sim(spacing):
    # Set up logging.
    # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # logdir = 'logs\\simple_angular_prop_tf\\%s' % stamp
    # writer = tf.summary.create_file_writer(logdir)
    # tf.summary.trace_on(graph=True, profiler=True)

    # define the datatype to be used in this simulation
    # The complex datatype must have double the bits of the float or you'll get casting errors (since a complex is
    # just 2 floats)
    # using float32/complex64 was a little bit faster than float64/complex128
    dtype = {'real': tf.float32, 'comp': tf.complex64, }

    sim_args = {
        'wavelength': 633e-9,  # HeNe
        # 'slm_size': 473,  # defines a simulation region of `slm_size` by `slm_size`
        'slm_size': 401,  # defines a simulation region of `slm_size` by `slm_size`
    }

    sim_args = {
        **sim_args,
        'lens_aperture': 2e-3,
    }

    sim_args = {
        **sim_args,
        **{
            'k': 2. * np.pi / sim_args['wavelength'],

            #'spacing_1d_to_ms1': sim_args['lens_aperture']/sim_args['wavelength']*8e-6, #NA so d_min is lambda
            'spacing_1d_to_ms1': 2e-3,
            'spacing_ms1_to_ms2': 1.5 * 1e-3, # thickness of glass substrate
            'spacing_ms2_to_detector': 20e-3,


            'slm_shape': (sim_args['slm_size'], sim_args['slm_size'],),
            'dd': sim_args['lens_aperture'] / sim_args['slm_size'],  # array element spacing
            # allows `filter_height` * `filter_width` number of orthogonal modes through
            # 'filter_height': 7,
            # 'filter_width': 7,
            'n_i_bins': 2,
            'n_j_bins': 2,

            'n_modes': 4,
        }
    }

    # initialize the 1D SLM basis set.
    # 1D SLM pixel coeffients
    # weighs is a constant TODO: Make constants
    weights = tf.Variable(tf.ones(sim_args['n_modes'], dtype=dtype['comp']))


    slm_args = {
        'n_weights': sim_args['n_modes'],
        'pixel_width': 2,
        'pixel_height': 2,
        'pixel_spacing': spacing,
        'dtype': dtype['comp']}
    slm_args['end_spacing'] = (sim_args['slm_size'] - (
            sim_args['n_modes']*slm_args['pixel_height'] + (sim_args['n_modes'] - 1) * slm_args['pixel_spacing']
    )) // 2

    assert slm_args['end_spacing'] >= 0, "Bounds error"

    logging.warning(slm_args)
    logging.warning(sim_args)

    field_generator = oneD_slm_field_generator.OneDPhasorField(**slm_args)

    # Make sure that this input modes shape match the simulation shape
    assert field_generator.n == sim_args['slm_size'], "SLM field and simulation field mismatch. Adjust the 1D Slm structure. ({} " \
                                         "vs {})".format(field_generator.n, sim_args['slm_size'])


    metasurface1_phase = tf.Variable(
        tf.random.uniform(shape=sim_args['slm_shape'], maxval=np.pi*2., dtype=dtype['real']),
        trainable=True,
    )
    metasurface2_phase = tf.Variable(
        tf.random.uniform(shape=sim_args['slm_shape'], maxval=np.pi*2., dtype=dtype['real']),
        trainable=True,
    )

    #
    # # initialize and define first metasurface
    # metasurface1_real, metasurface1_imag = split_complex(complex_initializer_random(shape=sim_args['slm_shape']))
    # metasurface1_real = tf.Variable(tf.cast(metasurface1_real, dtype=dtype['real']), dtype=dtype['real'],
    #                                 trainable=True)
    # metasurface1_imag = tf.Variable(tf.cast(metasurface1_imag, dtype=dtype['real']), dtype=dtype['real'],
    #                                 trainable=True)
    #
    # # initialize and define second metasurface
    # metasurface2_real, metasurface2_imag = split_complex(complex_initializer_random(shape=sim_args['slm_shape']))
    # metasurface2_real = tf.Variable(tf.cast(metasurface2_real, dtype=dtype['real']), dtype=dtype['real'],
    #                                 trainable=True)
    # metasurface2_imag = tf.Variable(tf.cast(metasurface2_imag, dtype=dtype['real']), dtype=dtype['real'],
    #                                 trainable=True)


    optimizer = tf.optimizers.Adam(learning_rate=0.1)
    # Training loop
    iterations = 2 ** 8
    n_update = 2 ** 4  # Prints information every `n_update` iterations

    # Training loop
    t = time.time()
    for i in range(iterations):
        error = train_step(
            metasurface1_phase, metasurface2_phase, weights, field_generator, sim_args, optimizer
        )
        if i % n_update == 0:
            t_now = time.time()
            print("Error: {}\tTimePerUpdate(s): {}\t {}/{}".format(error, t_now - t, i + 1, iterations))
            t = t_now

    metasurface1_real, metasurface1_imag = tf.math.cos(metasurface1_phase), tf.math.sin(metasurface1_phase)
    metasurface2_real, metasurface2_imag = tf.math.cos(metasurface2_phase), tf.math.sin(metasurface2_phase)



    plotting=False
    if plotting:
        figure_dir = os.path.join("..", "figures")
        assert os.path.isdir(figure_dir), "Folder '{}' doesn't exist".format(figure_dir)

        # Phase plot should use a cyclic color map: ['twilight', 'twilight_shifted', 'hsv']


        # Simulation complete. Now plotting results.
        fields = field_generator(weights)
        plot_slice(
            tf.abs(tf.reduce_sum(fields, axis=0)).numpy(),
            title="",
            sim_args=sim_args
        )
        plt.colorbar()
        plt.set_cmap('magma')
        plt.savefig(os.path.join(figure_dir, "input_SLM.pdf"), format='pdf', dpi=1000)
        plt.show()
        #
        # # plot individual incoming modes
        # plot_modes(fields, 5)
        # plt.set_cmap('jet')
        # plt.savefig(os.path.join(figure_dir, "incoming_modes.pdf"), format='pdf', dpi=1000)
        # plt.show()

        # plot ms1 phase
        plot_slice(
            np.angle(metasurface1_real.numpy() + 1j * metasurface1_imag.numpy()),
            "",
            sim_args,
        )
        plt.set_cmap('twilight')
        plt.colorbar()
        plt.savefig(os.path.join(figure_dir, "ms1.pdf"), format='pdf', dpi=1000)
        plt.show()

        # plot ms2 phase
        plot_slice(
            np.angle(metasurface2_real.numpy() + 1j * metasurface2_imag.numpy()),
            "",
            sim_args,
        )
        plt.set_cmap('twilight')
        plt.colorbar()
        plt.savefig(os.path.join(figure_dir, "ms2.pdf"), format='pdf', dpi=1000)
        plt.show()

        plot_modes(forward(weights, field_generator, sim_args, metasurface1_phase, metasurface2_phase).numpy(), 5)
        plt.set_cmap('magma')
        plt.savefig(os.path.join(figure_dir, "propagated_modes.pdf"), format='pdf', dpi=1000)
        plt.show()
        plt.set_cmap('magma')

        # plot_modes_fft(forward(weights, field_generator, sim_args, metasurface1_phase, metasurface2_phase))
        # plt.show()

        # plot abs correlation matrix
        plt.figure()
        plt.imshow(
            np.abs(
                correlation_matrix(forward(weights, field_generator, sim_args, metasurface1_phase, metasurface2_phase))
            )
        )
        plt.savefig(os.path.join(figure_dir, "correlation.pdf"), format='pdf', dpi=1000)
        plt.colorbar()
        plt.show()



    return np.abs(
                correlation_matrix(forward(weights, field_generator, sim_args, metasurface1_phase, metasurface2_phase))
    )


if __name__ == '__main__':
    correlation_mat = []
    for spacing in range(1, 101, 2):
        correlation_mat.append((spacing, run_sim(spacing)))

    print(correlation_mat)

    pickle.dump(correlation_mat, open("pitch_sweep.p", "wb"))
    # data = pickle.load(open("pitch_sweep.p", "rb"))



    def cross_corr(mat):
        n = np.shape(mat)[0]
        indicies = strictly_triangular_indices(n)

        cross = 0
        for i in indicies:
            cross += mat[i]
        return cross


    spacings = []
    sum_cross_coors = []
    for (spacing, mat) in correlation_mat:
        spacings.append(spacing)
        sum_cross_coors.append(cross_corr(mat))

    plt.scatter(spacings, sum_cross_coors)
    plt.show()