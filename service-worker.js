if(!self.define){let e,s={};const i=(i,a)=>(i=new URL(i+".js",a).href,s[i]||new Promise((s=>{if("document"in self){const e=document.createElement("script");e.src=i,e.onload=s,document.head.appendChild(e)}else e=i,importScripts(i),s()})).then((()=>{let e=s[i];if(!e)throw new Error(`Module ${i} didn’t register its module`);return e})));self.define=(a,r)=>{const n=e||("document"in self?document.currentScript.src:"")||location.href;if(s[n])return;let o={};const t=e=>i(e,n),d={module:{uri:n},exports:o,require:t};s[n]=Promise.all(a.map((e=>d[e]||t(e)))).then((e=>(r(...e),o)))}}define(["./workbox-d249b2c8"],(function(e){"use strict";self.addEventListener("message",(e=>{e.data&&"SKIP_WAITING"===e.data.type&&self.skipWaiting()})),e.precacheAndRoute([{url:"about/index.html",revision:"d1378ae835d43971259c76afeddeb1ea"},{url:"archives/2021/08/index.html",revision:"9543ce6f7733a2993994fd470c78404a"},{url:"archives/2021/index.html",revision:"d0989b1222bf74d54e5e5e8a280970c7"},{url:"archives/2022/11/index.html",revision:"9a79b24f712c546ab99ae3382ad408af"},{url:"archives/2022/index.html",revision:"beaf101d97aa80d02ec8af53e2ba10ce"},{url:"archives/index.html",revision:"efdeff41ae28af059ba3a9affd89c35d"},{url:"assets/css/style.css",revision:"8c3757344b6b13071c5d05c4af8c7b7c"},{url:"assets/posts/Arc-Browser/arc-demo.png",revision:"6827be6be2a7fad0b8b5d050365e0e33"},{url:"assets/posts/Arc-Browser/arc-sidebar.png",revision:"c3ac1c713fa8a7f339d49d07215a8594"},{url:"assets/posts/Arc-Browser/library.gif",revision:"1e9b61b1cf3afbf9f49c1c63c0212fd5"},{url:"assets/posts/Arc-Browser/test.gif",revision:"c626814c61077cffbb3d130e8ebf7df5"},{url:"assets/posts/Arc-Browser/wazenmai_Arc.png",revision:"d9f9b56cf3a643d6c44b175d0d0fe408"},{url:"assets/posts/hello-world/asexiaul.png",revision:"c2c2088ab1cbd774faa7af7611c3ba06"},{url:"assets/posts/MIDI-BERT/midibert.png",revision:"5da52842197c7fb6da7698f9d5def800"},{url:"assets/posts/MIDI-BERT/top_img.png",revision:"513d0966e4e502a1ec7063ea422c320a"},{url:"assets/posts/ProDA/ablation.png",revision:"4bd323059dad34f5d601aa39ba4c7ea0"},{url:"assets/posts/ProDA/GTA5.png",revision:"b946f496f744b1bf7670661981a3558c"},{url:"assets/posts/ProDA/Issue.png",revision:"d650457b4aa98503367cfb813b917937"},{url:"assets/posts/ProDA/model1.png",revision:"156a0aece175d4a09006ea0dd10156af"},{url:"assets/posts/ProDA/model2.png",revision:"ceb6d0fc739af1aecea1716e3bb3d7ef"},{url:"assets/posts/ProDA/model3.png",revision:"bf06086e0193abf137ac52d4f3703c08"},{url:"assets/posts/ProDA/ProDA.png",revision:"352291ff29e29f6cd05e13539e6e9209"},{url:"assets/posts/ProDA/prototype.png",revision:"c1de210fabeab132d328db5af38ed34e"},{url:"assets/posts/ProDA/Synthia.png",revision:"dd83f8b2ce96753e5d587fe80bcb7ea8"},{url:"assets/posts/ProDA/umap.png",revision:"1eafbff90f921f574e0fc93d1624b288"},{url:"assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/CC_NTHU.png",revision:"102a9e77e939963c869469d5f180850e"},{url:"assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/GTA5_CC.png",revision:"b1bccc58571d2cebbe4f676b6d7368c3"},{url:"assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/model.png",revision:"7f06b8e84b65d8e4e3bc0764805a2c17"},{url:"assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/Syn_CC.png",revision:"0721bb6be9270b3f36d999d4d0d5db22"},{url:"assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/toy_example.png",revision:"160e540db03d05ba99505cf028cde877"},{url:"assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/toy_problem.png",revision:"2dc876fc57cb0b1d1a7b231b0aab657d"},{url:"assets/profile.jpg",revision:"fc93d378e50a7fde556e4ff2b02589c2"},{url:"assets/top_img.png",revision:"e438f889cebec6a1d226a8aa7d4bffef"},{url:"categories/index.html",revision:"2e03be355a4392241f0aa4d8606da414"},{url:"categories/Machine-Learning/index.html",revision:"7f89b56599e13feeba1e118eab368e07"},{url:"categories/好物推薦/index.html",revision:"0a88bf32bfa5b98a360376d2a9ffefdd"},{url:"css/index.css",revision:"37c9252ac2fb22d06f5768728cd9190d"},{url:"css/var.css",revision:"d41d8cd98f00b204e9800998ecf8427e"},{url:"img/404.jpg",revision:"4ef3cfb882b6dd4128da4c8745e9a507"},{url:"img/favicon.png",revision:"7a8c47cb5a2149c1a1af21e90ecd9ca7"},{url:"img/friend_404.gif",revision:"68af0be9d22722e74665ef44dd532ba8"},{url:"index.html",revision:"cd641cd5e487fdb5cd07dd13b54b16ef"},{url:"js/main.js",revision:"b28662bf81abe2838d20faf2980f0034"},{url:"js/search/algolia.js",revision:"308de383b932f321e04751fd1f79cffb"},{url:"js/search/local-search.js",revision:"149fcc60c1de0a818e111978d01cbd87"},{url:"js/tw_cn.js",revision:"58cb9443474ea5d84a1c4ac806c8b4b7"},{url:"js/utils.js",revision:"24971090b8b1bd5d3f538d414e270fd3"},{url:"music/index.html",revision:"8bc4c2a29e6d49e091c2fe19395bbd22"},{url:"tags/Recommandation/index.html",revision:"2392be707801f70b8b719cb682906c93"},{url:"tags/Semantic-Segmentation/index.html",revision:"feb6e238d7e45f9ccf1aa76f4e162ff5"},{url:"tags/UDA/index.html",revision:"a6dab7151378a1f14ce03a51e66b20f2"},{url:"好物推薦-Arc-Browser/index.html",revision:"0010d50deb4448c1e3cf258a00ec0575"},{url:"論文筆記-ProDA-Prototypical-Pseudo-Label-Denoising-and-Target-Structure-Learning-for-Domain-Adaptive-Semantic-Segmentation/index.html",revision:"320f29689da7207fa52ba4388f93635b"},{url:"論文筆記-Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/index.html",revision:"d8c767d45f1fa6a1b87765adb93c591c"}],{})}));
//# sourceMappingURL=service-worker.js.map
