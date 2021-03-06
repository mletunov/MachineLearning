
import constants from '../constants';
var agent = require('superagent-promise')(require('superagent'), Promise);


let commonAPI = {
    uploadFiles(files, callback) {
        var req = agent.post(`${constants.API}upload`);
      files.forEach((file)=> {
          req.attach('videoFile', file);
      });
      return req.end();
  },

  getSessionData(sessionId){
      return fetch(`${constants.API}session/${sessionId}`, {
      method: 'get',
      headers: {'Content-Type': 'application/json',}
    })
    .then((response) => response.json())
  }
};

export default commonAPI;
