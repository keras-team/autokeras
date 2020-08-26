const { Octokit } = require("@octokit/rest");
const octokit = new Octokit({});

var width = 890;
var height = 198;
var size = 36;
var gap = 3;
var elem_per_line = Math.floor(width / (size + gap));

octokit.paginate(octokit.repos.listContributors,{
    owner: 'keras-team',
    repo: 'autokeras',
    per_page: 100,
    anon: 1,
}).then((contributors) => {
    //console.log(contributors);
    html = buildHTML(contributors);
    console.log(html);
});

function buildHTML(contributors) {
    html = `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${width}" height="${height}">`;

    defs = '<defs>'
    defs += '<rect id="rect" width="36" height="36" rx="18"/>';
    defs += '<clipPath id="clip"> <use xlink:href="#rect"/> </clipPath> ';
    defs += '</defs>'

    html += defs

    for (i = 0; i < contributors.length; i++) {
        xi = i % elem_per_line;
        yi = Math.floor(i / elem_per_line);
        x = xi * (size + gap) + gap;
        y = yi * (size + gap) + gap;
        temp = `<a xlink:href="${contributors[i].html_url} "><image transform="translate(${x},${y})" xlink:href="${contributors[i].avatar_url}" alt="${contributors[i].login}" clip-path="url(#clip)" width="36" height="36"/></a>`;
        html += temp;
    }

    html += '</svg>';
    return html;
}
