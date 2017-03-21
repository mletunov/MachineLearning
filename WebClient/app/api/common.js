
import constants from '../constants';
var agent = require('superagent-promise')(require('superagent'), Promise);


let commonAPI = {
    uploadFiles(files, callback) {
        var req = agent.post(`${constants.API}/upload`);
      files.forEach((file)=> {
          req.attach('videoFile', file);
      });
      return req.end();
  },
  
  getTimeStamps(fileName){
      return fetch(`${constants.API}/getTimeStamps`, {
      method: 'post',
      headers: {'Content-Type': 'application/json',},
      body: JSON.stringify({fileName})
    })
    .then((response) => response.json())
  }
};

export default commonAPI;
