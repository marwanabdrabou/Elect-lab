let menu=document.querySelector('#menu-btn');
let navbar=document.querySelector('.navbar');

menu.onclick = () =>{
  menu.classList.toggle('fa-times');
  navbar.classList.toggle('active');
}

window.onscroll = () =>{
  menu.classList.remove('fa-times');
  navbar.classList.remove('active');
}





// upload photo start


const dragArea= document.querySelector('.drag-area');
const dragText = document.querySelector('.head');

let button = document.querySelector('.button');
let input = document.querySelector('.browse');

let file;

button.onclick = () => {
  input.click();
};

// when click on browse
input.addEventListener('change' , function () {
  file = this.files[0];
  dragArea.classList.add('active');
  displayFile();
})


// when file is inside drag area
dragArea.addEventListener('dragover', (event) => {
  event.preventDefault();
  dragText.textContent = 'Release to Upload';
  dragArea.classList.add('active');


  // console.log('file is inside drag area');
})


// when file leave drag area
dragArea.addEventListener('dragleave',() =>{
  dragText.textContent = 'Drag & Drop';
  dragArea.classList.remove('active');


  // console.log('file is left drag area');
})

// when file dropped in drag area
dragArea.addEventListener('drop', (event) => {
  event.preventDefault();

  file = event.dataTransfer.files[0];       //choose first file only if they are multiple files
  //console.log(file);
});

function displayFile() {
  let fileType = file.type;                // preview the file type
  console.log(fileType);

  let validExtension = ['image/png' , 'image/jpg' , 'image/jpeg'];
  if(validExtension.includes(fileType)) {
    let fileReader = new FileReader();

    fileReader.onload = () => {
        let fileURL = fileReader.result;
        //console.log(fileURL);
        let imgTag = `<img src ="${fileURL}" alt="">`;
        dragArea.innerHTML = imgTag;
    };
    fileReader.readAsDataURL(file);

  } else {
    alert('This file is not an Image');
    dragArea.classList.remove('active');
  }
  // console.log('file dropped in drag area');
}


// upload photo end

