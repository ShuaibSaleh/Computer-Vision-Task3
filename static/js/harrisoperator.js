let image_input = document.querySelector('#imageinput');
let imageEdges = document.querySelector('#imageEdges');
let edgedetect_method=document.querySelector("#edgedetectmethod");
let submit=document.querySelector("#submit_btn");
let image_data = ""




image_input.addEventListener('change', e => {
    if (e.target.files.length) {
      const reader = new FileReader();
      reader.onload = e => {
        if (e.target.result) {
          let img = document.createElement('img');
          img.id ="imageEdges";
          img.src = e.target.result;
          imageEdges.innerHTML = '';
          imageEdges.appendChild(img)
          image_data = e.target.result
  
          
        }
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  });




submit.addEventListener('click', e => {
e.preventDefault();
send();
}
)

    
function send(){
        
      let formData = new FormData();

      try {
        // console.log("done1");

       if (image_data == "") {
        throw "error : not enought images "
      }
      // console.log("done2");
      formData.append('image_data',image_data);
      // formData.append("edgedetectmethod" ,edgedetect_method.value);
      // console.log("done3");
    
      console.log("formdata done")
      $.ajax({
        type: 'POST',
        url: '/harris',
        data: formData,
        cache: false,
        contentType: false,
        processData: false,
        async: true,
        success: function (backEndData) {
          console.log("done done done 1");
          var responce = JSON.parse(backEndData)
          console.log(responce)
          console.log(responce[1])
          console.log(responce[2])
          // console.log(responce[3])

          let ApplyEdges = document.getElementById("ApplyEdges")
          let info1 = document.getElementById("info1")
          // let info2 = document.getElementById("info2")
          ApplyEdges.remove()
          info1.remove()
          // info2.remove()
          ApplyEdges = document.createElement("div")
          info1 = document.createElement("div")
          // info2 = document.createElement("div")
          ApplyEdges.id = "ApplyEdges"
          info1.id = "info1"
          // info2.id = "info1"
          ApplyEdges.innerHTML = responce[1]
          info1.innerHTML = responce[2]
          // info2.innerHTML = responce[3]
          
          
          let col2 = document.getElementById("Col2")
          col2.appendChild(ApplyEdges)
          let mycalc = document.getElementById("mycalc")
          mycalc.appendChild(info1)
          // mycalc.appendChild(info2)
          console.log("done done done 2");
        }
      })
      console.log("ajax done")
      console.log("Please Wait, The harris image will apear now")
    }
     catch (error) {
      console.log("please upload the image")
    } 
  }