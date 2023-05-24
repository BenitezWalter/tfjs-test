const inpEpoca = document.getElementById("inpEpoca");
const inpX = document.getElementById("inpX");

const losses = [];

const model = tf.sequential();

const fit = async () => {
  const epocas = parseInt(inpEpoca.value);

  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  //Formula: Y = 2x - 3
  const xs = tf.tensor2d(
    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [17, 1]
  );

  const ys = tf.tensor2d(
    [-13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    [17, 1]
  );

  const lossContainer = document.getElementById("loss-container");
  const metrics = ["loss"];

  const metricsContainer = {
    name: "Metrics",
    tab: "Charts",
    styles: { height: "100px" },
  };

  const lossChart = tfvis.show.fitCallbacks(
    lossContainer,
    metrics,
    metricsContainer
  );

  await model.fit(xs, ys, {
    epochs: epocas,
    callbacks: lossChart,
  });

  document.getElementById("res").innerText = "Entrenamiento terminado";
};

const predict = () => {
  const inp = parseInt(inpX.value);
  document.getElementById("micro-out-div").innerText = model
    .predict(tf.tensor2d([inp], [1, 1]))
    .dataSync();
};
