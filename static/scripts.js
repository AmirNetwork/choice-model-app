// static/scripts.js
document.addEventListener('DOMContentLoaded', function() {
  const queryForm = document.getElementById("queryForm");
  const resultDiv = document.getElementById("result");
  const flashMessagesDiv = document.getElementById("flash-messages");

  if(queryForm){
    queryForm.addEventListener("submit", function(e){
      e.preventDefault();
      const formData = new FormData(queryForm);

      fetch("/", {
        method: "POST",
        body: formData
      })
      .then(response => response.text())
      .then(html => {
        // Parse the returned HTML
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');

        // Update flash messages
        const newFlash = doc.querySelector('#flash-messages');
        if(newFlash && flashMessagesDiv){
          flashMessagesDiv.innerHTML = newFlash.innerHTML;
        }

        // Update result
        const newResult = doc.querySelector('#result');
        if(newResult && resultDiv){
          resultDiv.innerHTML = newResult.innerHTML;
        }
      })
      .catch(err => console.error("Error with fetch:", err));
    });
  }
});
