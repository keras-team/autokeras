const { Octokit } = require("@octokit/rest");
const fs = require('fs')
const octokit = new Octokit({});

octokit.paginate(octokit.repos.listContributors,{
    owner: 'keras-team',
    repo: 'autokeras',
}).then((contributors) => {
    fs.writeFileSync('contributors.json', JSON.stringify(contributors))
});
