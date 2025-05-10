// Ensure Dropzone auto discovery is disabled if you plan to use a custom file upload approach
Dropzone.autoDiscover = false;

// Initialize Dropzone (optional, depending on your frontend design)
let fileUploadZone = new Dropzone("#dropzone", {
    maxFiles: 1,
    acceptedFiles: ".jpg,.jpeg,.png"
});

// Handle the form submission via jQuery
$('#upload-form').submit(function(e) {
    e.preventDefault();  // Prevent default form submission

    var formData = new FormData(this);  // Create a FormData object to hold the file

    // Make the AJAX request to the Flask backend
    $.ajax({
        url: '/predict',  // This should match your Flask route
        type: 'POST',
        data: formData,
        success: function(data) {
            if (data.error) {
                $('#result').text('Error: ' + data.error);  // Handle error response
            } else {
                // Display the uploaded image and prediction result
                $('#result').html(`<img src="${data.image_path}" class="img-fluid" />`);
                $('#result').append(`<p>Prediction: ${data.prediction}</p>`);
            }
        },
        cache: false,
        contentType: false,
        processData: false
    });
});
