let imageA_input = document.querySelector('#imageA_input')
let imageB_input = document.querySelector('#imageB_input')
let imageA = document.querySelector('#imageA')
let imageB = document.querySelector('#imageB')
let submit = document.getElementById("submit_btn")

let imageA_data = ""
let imageB_data = ""

let method = document.getElementById("method")






imageA_input.addEventListener('change', e => {
    if (e.target.files.length) {
        // start file reader
      const reader = new FileReader();
      reader.onload = e => {
        if (e.target.result) {
          // create new image
          let img = document.createElement('img');
          img.id = 'imageA';
          img.src = e.target.result;
          // clean result before
          imageA.innerHTML = '';
          // append new image
          imageA.appendChild(img)
          // origial image
          imageA_data = e.target.result

        }
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  });

imageB_input.addEventListener('change', e => {
    if (e.target.files.length) {
      // start file reader
      const reader = new FileReader();
      reader.onload = e => {
        if (e.target.result) {
          // create new image
          let img = document.createElement('img');
          img.id = 'imageB';
          img.src = e.target.result;
          // clean result before
          imageB.innerHTML = '';
          // append new image
          imageB.appendChild(img)
          // origial image
          imageB_data = e.target.result
        }
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  });

function send(){
  
    // to handle if the user not enter two images
    try {
      const option = document.getElementById("image1_info");
      if (imageA_data == "" || imageB_data == "") {
        throw "error : not enought images "
      }

      let formData = new FormData();
      formData.append('imageA_data',imageA_data)
      formData.append('imageB_data',imageB_data)
      formData.append('method',method.value)
      
      
    
      $.ajax({
        type: 'POST',
        url: '/SSD_NCC_match',
        data: formData,
        cache: false,
        contentType: false,
        processData: false,
        async: true,
        success: function (backEndData) {
          var responce = JSON.parse(backEndData)

          let imageC = document.getElementById("imageC")
          let info1 = document.getElementById("info1")
          imageC.remove()
          info1.remove()
          imageC = document.createElement("div")
          info1 = document.createElement("div")
          imageC.id = "imageC"
          info1.id = "info1"
          imageC.innerHTML = responce[1]
          info1.innerHTML = responce[2]

          let imagediv3 = document.getElementById("imagediv3")
          
          imagediv3.appendChild(imageC)
          imagediv3.appendChild(info1)
          
  
        }
  
  
      })
    } catch (error) {
      console.log("please upload two images")
    } 
  }


submit.addEventListener('click', e => {
    e.preventDefault();
    send()
  }
  )