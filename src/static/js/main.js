//max should be 50

var value_1 = 49;
var value_2 =49;
var value_3 = 49;

window.onload = function() {
//    jasmineFunction = function() {//var value=20;
//    value+=0.1;
    var percentage_1a=value_1*2;
    var percentage_1b=(value_1*2)+5;
    var gradient_1="-webkit-linear-gradient(left,rgba(237,24,70,1) "+percentage_1a+"%,rgba(63,169,245,1)"+percentage_1b+"%)";
    
    var percentage_2a=value_2*2;
    var percentage_2b=(value_2*2)+5;
//    var gradient_2="-webkit-linear-gradient(left,rgba(87,202,238,1) "+percentage_2a+"%,rgba(241,90,36,1)"+percentage_2b+"%)";

    
  var gradient_2="-webkit-linear-gradient(left,rgba(241,90,36,1) "+percentage_2a+"%,rgba(128, 218, 219,1)"+percentage_2b+"%)";
    
    var percentage_3a=value_3*2;
    var percentage_3b=(value_3*2)+5;
    var gradient_3="-webkit-linear-gradient(left,rgba(34,181,115,1) "+percentage_3a+"%,rgba(102,45,145,1) "+percentage_3b+"%)";
        

    $("#continuum-1").css({
        background:gradient_1
    })
    
    $(".pointer-container").css({
        left:percentage_1a+"%"
    })
    
    $(".continuumValue").css({
        left:percentage_1a+"%"
    })
    
    
    $("#continuum-2").css({
        background:gradient_2
    })
    
    $(".pointer-container-2").css({
        left:percentage_2a+"%"
    })
    
    $(".continuumValue-2").css({
        left:percentage_2a+"%"
    })
    
    $("#continuum-3").css({
        background:gradient_3
    })
    
    $(".pointer-container-3").css({
        left:percentage_3a+"%"
    })
    
    $(".continuumValue-3").css({
        left:percentage_3a+"%"
    })
//                                     }
    
//    var _jasmineFunction = function() {
//        for (var i=0;i<50;i++) {
//            setTimeout(jasmineFunction, 1000)
//            
//            
//        }
//        
//    }
//    _jasmineFunction();
    
//    setTimeout(jasmineFunction(i), 3000)
 
}