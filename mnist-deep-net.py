import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def model_neural_network(data, num_nodes_list):
    num_layers = len(num_nodes_list)
    previous_layer_output = data

    for lnum in range(num_layers - 1):
        if lnum == num_layers - 2:
            is_output_layer = True
        else:
            is_output_layer = False

        num_nodes = num_nodes_list[lnum]
        num_nodes_next = num_nodes_list[lnum + 1]

        weights = tf.Variable(tf.random_normal([num_nodes, num_nodes_next]))
        biases = tf.Variable(tf.random_normal([num_nodes_next]))

        layer = tf.matmul(previous_layer_output, weights) + biases

        if not is_output_layer:
            layer = tf.nn.relu(layer)

        previous_layer_output = layer

    return previous_layer_output

def train_neural_network(batch_size, mnist, layer_sizes):
    input_layer_size = layer_sizes[0]
    x = tf.placeholder('float', [None, input_layer_size])
    y = tf.placeholder('float')

    prediction = model_neural_network(x, layer_sizes)
    cost_func = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    cost = tf.reduce_mean(cost_func)

    # default learning_rate = 0.001
    optimizer= tf.train.AdamOptimizer().minimize(cost)

    # cycles of FF + BP
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        ### Training ###
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss', epoch_loss)
        ### Training ###


        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


batch_size = 100
n_classes = 10
image_height = 28 # MNIST images are 28x28 square
input_layer_nodes = image_height * image_height
layer_sizes = [input_layer_nodes, 500, 500, 500, n_classes]

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
train_neural_network(batch_size, mnist, layer_sizes)
