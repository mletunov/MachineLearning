toastr.options.closeButton = true;
toastr.options.closeMethod = 'fadeOut';
toastr.options.closeDuration = 450;
toastr.options.closeEasing = 'swing';

let logSrv = {
  Error: function(){
    toastr.error('Please try again later.', 'We are sorry, but something went wrong.')
  }
}

export {logSrv};
