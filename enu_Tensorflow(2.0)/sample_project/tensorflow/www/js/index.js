

$(document).on("click",'#registBtn',function(){ 
	
    location="register.html";

});
$(document).on("click",'#loginBtn',function(){ 
	login();
});

window.onkeydown = function() {
  var agt = navigator.userAgent.toLowerCase();
  if (agt.indexOf("chrome") != -1)
  {
	  if(arguments[0].keyCode == '13')
	  {
		  login();
	  }
  }
  //if (agt.indexOf("opera") != -1) return 'Opera';
  if (agt.indexOf("firefox") != -1)
  {
	  if(keyCode == '13')
	  {
		  login();
	  }
  }
  //if (agt.indexOf("safari") != -1) return 'Safari';
  //if (agt.indexOf("mozilla/5.0") != -1) return 'Mozilla';
  
  
}

function login() 
{
   var userid= document.getElementById("userid").value;
   var password = document.getElementById("password").value;

    var xmlHttp = new XMLHttpRequest();
   var strUrl = "login" ;
   var strParam= "userid="+userid + "&" + "password="+ password;  

   //////////////////////////////////////////////////////////////////////
   xmlHttp.onreadystatechange=function()
   {
      if (xmlHttp.readyState==4 && xmlHttp.status==200)
       {       
          var msg = xmlHttp.responseText;
   
         var arr = JSON.parse(msg);      
         if (arr.RESULT == "OK")
         {
            dlsplayCanvas(arr.INDEX_PICTURE);
         }
            else if(arr.RESULT == "FAIL")
            {
                var error_msg = document.getElementById("error_msg");
                error_msg.innerHTML = "invalid id or password";
			
            }
       }
    };
   //////////////////////////////////////////////////////////////////////
   
   xmlHttp.open("POST",strUrl,true);   
   xmlHttp.setRequestHeader("Content-Type","application/x-www-form-urlencoded;charset=UTF-8");
   xmlHttp.setRequestHeader("Cache-Control","no-cache, must-revalidate");
   xmlHttp.setRequestHeader("Pragma","no-cache");
   xmlHttp.send(strParam);   
}
function dlsplayCanvas(INDEX_PICTURE)
{
	$('.login-page').empty();
	$('.login-page').remove();
	$('#Canvas_cover').show();
	var page =INDEX_PICTURE[0];
	requestpage(page);
}

function registeruser()
{
	var userid= document.getElementById("userid").value;
	var name = document.getElementById("name").value;
	var password = document.getElementById("password").value;
	var confirmpw = document.getElementById("confirmpw").value;	
	
	if (password != confirmpw)
	{
		alert("confirm password worng.");
		return;
	}

	//////////////////////////////////////////////////////////////////////	
    var xmlHttp = new XMLHttpRequest();
	var strUrl = "registeruser" ;
	var strParam= "userid="+userid + "&" + "name="+ name + "&" + "password="+ password;  

	xmlHttp.onreadystatechange=function()
	{
		if (xmlHttp.readyState==4 && xmlHttp.status==200)
	    {	    
	    	var msg = xmlHttp.responseText;
	
			var arr = JSON.parse(msg);		
			if (arr.RESULT == "OK")
			{
				location = "index.html";
			}
			else
			{
				if (arr.RESULT_CODE == "CODE_EXIST_USER" )
				{
					alert("이미 등록된 사용자입니다.");
				}
				if (arr.RESULT_CODE == "CODE_EMPTY_PARAMETER" )
				{
					alert("입력 파라미터의 정보가 잘못되었습니다..");
				}				
			}
	    }
	 };
	//////////////////////////////////////////////////////////////////////
	
	xmlHttp.open("POST",strUrl,true);	
	xmlHttp.setRequestHeader("Content-Type","application/x-www-form-urlencoded;charset=UTF-8");
	xmlHttp.setRequestHeader("Cache-Control","no-cache, must-revalidate");
	xmlHttp.setRequestHeader("Pragma","no-cache");
	xmlHttp.send(strParam);	
}

function CanvasSetup()
{
    
    //브라우저 사이즈를 알아낸다.
    
     var Bwidth = document.body.clientWidth;
     var Bheight =window.innerHeight;
	
	 //
    //높이구하기
    //head 크기 알아냄
	//var canvas = $('#ID_CANVAS');
    canvas.height = Bheight;
    canvas.clientHeight = canvas.height;
    canvas.width = Bwidth -10;
    canvas.clientWidth =  canvas.width;
    
	var deltaY =document.body.scrollHeight -canvas.clientHeight;
	canvas.height = canvas.height -deltaY;
	canvas.clientHeight = canvas.height;
	
	ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
	ctx.rect(0,0,canvas.clientWidth,canvas.clientHeight);
	var worldSizeX =canvas.clientWidth;
	var worldsizeY =canvas.clientHeight;
	canvas.width =  worldSizeX;
	canvas.height=  worldsizeY;
           
}


