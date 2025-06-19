Dropzone.autoDiscover = false;

const dropzone = new Dropzone("#dropzoneForm", {
  url: "/predict",
  paramName: "file",
  maxFiles: 1,
  acceptedFiles: ".jpg,.jpeg,.png",
  autoProcessQueue: false,
  thumbnailWidth: 200,
  thumbnailHeight: 200,
  resizeWidth: null,
  resizeHeight: null,
  addRemoveLinks: true,
  dictDefaultMessage: "Drop image here or click to upload",
  init: function () {
    this.on("addedfile", function (file) {
      if (this.files.length > 1) {
        this.removeFile(this.files[0]);
      }

      const icon = document.querySelector(".upload-icon");
      if (icon) icon.style.display = "none";

      document.getElementById("result").innerHTML = "";
      document.getElementById("classified-image").innerHTML = "";
    });
  }
});

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.innerText = message;
  toast.style.display = "block";

  // Hide toast after animation ends
  setTimeout(() => {
    toast.style.display = "none";
  }, 3000);
}

function showLoader() {
  document.getElementById("loader-overlay").style.display = "flex";
}

function hideLoader() {
  document.getElementById("loader-overlay").style.display = "none";
}

document.getElementById('classify-btn').addEventListener('click', function (e) {
  e.preventDefault();

  if (dropzone.files.length > 0) {
    showLoader();
    dropzone.processQueue();
  } else {
    showToast("Please upload an image first.");
  }
});

dropzone.on("success", function (file, response) {
  const resultDiv = document.getElementById("result");
  const imageDiv = document.getElementById("classified-image");

  resultDiv.innerHTML = "";
  imageDiv.innerHTML = "";

  hideLoader();

  if (response.status === 'fail') {
    showToast(response.message);
  } else {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    img.alt = "Classified Person";
    imageDiv.appendChild(img);

    const sorted = Object.entries(response.class_probabilities).sort((a, b) => b[1] - a[1]);
    const top3 = sorted.slice(0, 3);  // get top 3 only
    const topClass = top3[0][0];

    const title = document.createElement("h2");
    title.innerText = "Prediction: " + response.predicted_class;
    resultDiv.appendChild(title);

    top3.forEach(([label, prob]) => {
      const bar = document.createElement("div");
      bar.classList.add("bar");
      bar.style.width = "0%"; // animate from zero
      bar.style.backgroundColor = label === topClass ? "limegreen" : "crimson";
      bar.innerText = `${label}: ${(prob * 100).toFixed(2)}%`;
      resultDiv.appendChild(bar);

      setTimeout(() => {
        bar.style.width = (prob * 100) + "%";
      }, 50);
    });
  }

  dropzone.removeAllFiles(true);
  const icon = document.querySelector(".upload-icon");
  if (icon) icon.style.display = "block";
});
