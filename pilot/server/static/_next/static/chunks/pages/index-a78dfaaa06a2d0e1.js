(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[405],{57838:function(e,t,n){"use strict";n.d(t,{Z:function(){return o}});var i=n(67294);function o(){let[,e]=i.useReducer(e=>e+1,0);return e}},96074:function(e,t,n){"use strict";n.d(t,{Z:function(){return u}});var i=n(94184),o=n.n(i),a=n(67294),l=n(53124),r=n(14747),s=n(67968),c=n(45503);let d=e=>{let{componentCls:t,sizePaddingEdgeHorizontal:n,colorSplit:i,lineWidth:o}=e;return{[t]:Object.assign(Object.assign({},(0,r.Wf)(e)),{borderBlockStart:`${o}px solid ${i}`,"&-vertical":{position:"relative",top:"-0.06em",display:"inline-block",height:"0.9em",margin:`0 ${e.dividerVerticalGutterMargin}px`,verticalAlign:"middle",borderTop:0,borderInlineStart:`${o}px solid ${i}`},"&-horizontal":{display:"flex",clear:"both",width:"100%",minWidth:"100%",margin:`${e.dividerHorizontalGutterMargin}px 0`},[`&-horizontal${t}-with-text`]:{display:"flex",alignItems:"center",margin:`${e.dividerHorizontalWithTextGutterMargin}px 0`,color:e.colorTextHeading,fontWeight:500,fontSize:e.fontSizeLG,whiteSpace:"nowrap",textAlign:"center",borderBlockStart:`0 ${i}`,"&::before, &::after":{position:"relative",width:"50%",borderBlockStart:`${o}px solid transparent`,borderBlockStartColor:"inherit",borderBlockEnd:0,transform:"translateY(50%)",content:"''"}},[`&-horizontal${t}-with-text-left`]:{"&::before":{width:"5%"},"&::after":{width:"95%"}},[`&-horizontal${t}-with-text-right`]:{"&::before":{width:"95%"},"&::after":{width:"5%"}},[`${t}-inner-text`]:{display:"inline-block",padding:"0 1em"},"&-dashed":{background:"none",borderColor:i,borderStyle:"dashed",borderWidth:`${o}px 0 0`},[`&-horizontal${t}-with-text${t}-dashed`]:{"&::before, &::after":{borderStyle:"dashed none none"}},[`&-vertical${t}-dashed`]:{borderInlineStartWidth:o,borderInlineEnd:0,borderBlockStart:0,borderBlockEnd:0},[`&-plain${t}-with-text`]:{color:e.colorText,fontWeight:"normal",fontSize:e.fontSize},[`&-horizontal${t}-with-text-left${t}-no-default-orientation-margin-left`]:{"&::before":{width:0},"&::after":{width:"100%"},[`${t}-inner-text`]:{paddingInlineStart:n}},[`&-horizontal${t}-with-text-right${t}-no-default-orientation-margin-right`]:{"&::before":{width:"100%"},"&::after":{width:0},[`${t}-inner-text`]:{paddingInlineEnd:n}}})}};var m=(0,s.Z)("Divider",e=>{let t=(0,c.TS)(e,{dividerVerticalGutterMargin:e.marginXS,dividerHorizontalWithTextGutterMargin:e.margin,dividerHorizontalGutterMargin:e.marginLG});return[d(t)]},{sizePaddingEdgeHorizontal:0}),p=function(e,t){var n={};for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&0>t.indexOf(i)&&(n[i]=e[i]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var o=0,i=Object.getOwnPropertySymbols(e);o<i.length;o++)0>t.indexOf(i[o])&&Object.prototype.propertyIsEnumerable.call(e,i[o])&&(n[i[o]]=e[i[o]]);return n},u=e=>{let{getPrefixCls:t,direction:n,divider:i}=a.useContext(l.E_),{prefixCls:r,type:s="horizontal",orientation:c="center",orientationMargin:d,className:u,rootClassName:g,children:h,dashed:b,plain:f,style:x}=e,v=p(e,["prefixCls","type","orientation","orientationMargin","className","rootClassName","children","dashed","plain","style"]),y=t("divider",r),[S,$]=m(y),w=c.length>0?`-${c}`:c,j=!!h,z="left"===c&&null!=d,C="right"===c&&null!=d,_=o()(y,null==i?void 0:i.className,$,`${y}-${s}`,{[`${y}-with-text`]:j,[`${y}-with-text${w}`]:j,[`${y}-dashed`]:!!b,[`${y}-plain`]:!!f,[`${y}-rtl`]:"rtl"===n,[`${y}-no-default-orientation-margin-left`]:z,[`${y}-no-default-orientation-margin-right`]:C},u,g),N=a.useMemo(()=>"number"==typeof d?d:/^\d+$/.test(d)?Number(d):d,[d]),O=Object.assign(Object.assign({},z&&{marginLeft:N}),C&&{marginRight:N});return S(a.createElement("div",Object.assign({className:_,style:Object.assign(Object.assign({},null==i?void 0:i.style),x)},v,{role:"separator"}),h&&"vertical"!==s&&a.createElement("span",{className:`${y}-inner-text`,style:O},h)))}},75081:function(e,t,n){"use strict";n.d(t,{Z:function(){return S}});var i=n(94184),o=n.n(i),a=n(98423),l=n(67294),r=n(96159),s=n(53124),c=n(23183),d=n(14747),m=n(67968),p=n(45503);let u=new c.E4("antSpinMove",{to:{opacity:1}}),g=new c.E4("antRotate",{to:{transform:"rotate(405deg)"}}),h=e=>({[`${e.componentCls}`]:Object.assign(Object.assign({},(0,d.Wf)(e)),{position:"absolute",display:"none",color:e.colorPrimary,fontSize:0,textAlign:"center",verticalAlign:"middle",opacity:0,transition:`transform ${e.motionDurationSlow} ${e.motionEaseInOutCirc}`,"&-spinning":{position:"static",display:"inline-block",opacity:1},"&-nested-loading":{position:"relative",[`> div > ${e.componentCls}`]:{position:"absolute",top:0,insetInlineStart:0,zIndex:4,display:"block",width:"100%",height:"100%",maxHeight:e.contentHeight,[`${e.componentCls}-dot`]:{position:"absolute",top:"50%",insetInlineStart:"50%",margin:-e.spinDotSize/2},[`${e.componentCls}-text`]:{position:"absolute",top:"50%",width:"100%",paddingTop:(e.spinDotSize-e.fontSize)/2+2,textShadow:`0 1px 2px ${e.colorBgContainer}`,fontSize:e.fontSize},[`&${e.componentCls}-show-text ${e.componentCls}-dot`]:{marginTop:-(e.spinDotSize/2)-10},"&-sm":{[`${e.componentCls}-dot`]:{margin:-e.spinDotSizeSM/2},[`${e.componentCls}-text`]:{paddingTop:(e.spinDotSizeSM-e.fontSize)/2+2},[`&${e.componentCls}-show-text ${e.componentCls}-dot`]:{marginTop:-(e.spinDotSizeSM/2)-10}},"&-lg":{[`${e.componentCls}-dot`]:{margin:-(e.spinDotSizeLG/2)},[`${e.componentCls}-text`]:{paddingTop:(e.spinDotSizeLG-e.fontSize)/2+2},[`&${e.componentCls}-show-text ${e.componentCls}-dot`]:{marginTop:-(e.spinDotSizeLG/2)-10}}},[`${e.componentCls}-container`]:{position:"relative",transition:`opacity ${e.motionDurationSlow}`,"&::after":{position:"absolute",top:0,insetInlineEnd:0,bottom:0,insetInlineStart:0,zIndex:10,width:"100%",height:"100%",background:e.colorBgContainer,opacity:0,transition:`all ${e.motionDurationSlow}`,content:'""',pointerEvents:"none"}},[`${e.componentCls}-blur`]:{clear:"both",opacity:.5,userSelect:"none",pointerEvents:"none","&::after":{opacity:.4,pointerEvents:"auto"}}},"&-tip":{color:e.spinDotDefault},[`${e.componentCls}-dot`]:{position:"relative",display:"inline-block",fontSize:e.spinDotSize,width:"1em",height:"1em","&-item":{position:"absolute",display:"block",width:(e.spinDotSize-e.marginXXS/2)/2,height:(e.spinDotSize-e.marginXXS/2)/2,backgroundColor:e.colorPrimary,borderRadius:"100%",transform:"scale(0.75)",transformOrigin:"50% 50%",opacity:.3,animationName:u,animationDuration:"1s",animationIterationCount:"infinite",animationTimingFunction:"linear",animationDirection:"alternate","&:nth-child(1)":{top:0,insetInlineStart:0},"&:nth-child(2)":{top:0,insetInlineEnd:0,animationDelay:"0.4s"},"&:nth-child(3)":{insetInlineEnd:0,bottom:0,animationDelay:"0.8s"},"&:nth-child(4)":{bottom:0,insetInlineStart:0,animationDelay:"1.2s"}},"&-spin":{transform:"rotate(45deg)",animationName:g,animationDuration:"1.2s",animationIterationCount:"infinite",animationTimingFunction:"linear"}},[`&-sm ${e.componentCls}-dot`]:{fontSize:e.spinDotSizeSM,i:{width:(e.spinDotSizeSM-e.marginXXS/2)/2,height:(e.spinDotSizeSM-e.marginXXS/2)/2}},[`&-lg ${e.componentCls}-dot`]:{fontSize:e.spinDotSizeLG,i:{width:(e.spinDotSizeLG-e.marginXXS)/2,height:(e.spinDotSizeLG-e.marginXXS)/2}},[`&${e.componentCls}-show-text ${e.componentCls}-text`]:{display:"block"}})});var b=(0,m.Z)("Spin",e=>{let t=(0,p.TS)(e,{spinDotDefault:e.colorTextDescription,spinDotSize:e.controlHeightLG/2,spinDotSizeSM:.35*e.controlHeightLG,spinDotSizeLG:e.controlHeight});return[h(t)]},{contentHeight:400}),f=function(e,t){var n={};for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&0>t.indexOf(i)&&(n[i]=e[i]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var o=0,i=Object.getOwnPropertySymbols(e);o<i.length;o++)0>t.indexOf(i[o])&&Object.prototype.propertyIsEnumerable.call(e,i[o])&&(n[i[o]]=e[i[o]]);return n};let x=null,v=e=>{let{spinPrefixCls:t,spinning:n=!0,delay:i=0,className:c,rootClassName:d,size:m="default",tip:p,wrapperClassName:u,style:g,children:h,hashId:b}=e,v=f(e,["spinPrefixCls","spinning","delay","className","rootClassName","size","tip","wrapperClassName","style","children","hashId"]),[y,S]=l.useState(()=>n&&(!n||!i||!!isNaN(Number(i))));l.useEffect(()=>{if(n){var e;let t=function(e,t,n){var i,o=n||{},a=o.noTrailing,l=void 0!==a&&a,r=o.noLeading,s=void 0!==r&&r,c=o.debounceMode,d=void 0===c?void 0:c,m=!1,p=0;function u(){i&&clearTimeout(i)}function g(){for(var n=arguments.length,o=Array(n),a=0;a<n;a++)o[a]=arguments[a];var r=this,c=Date.now()-p;function g(){p=Date.now(),t.apply(r,o)}function h(){i=void 0}!m&&(s||!d||i||g(),u(),void 0===d&&c>e?s?(p=Date.now(),l||(i=setTimeout(d?h:g,e))):g():!0!==l&&(i=setTimeout(d?h:g,void 0===d?e-c:e)))}return g.cancel=function(e){var t=(e||{}).upcomingOnly;u(),m=!(void 0!==t&&t)},g}(i,()=>{S(!0)},{debounceMode:!1!==(void 0!==(e=({}).atBegin)&&e)});return t(),()=>{var e;null===(e=null==t?void 0:t.cancel)||void 0===e||e.call(t)}}S(!1)},[i,n]);let $=l.useMemo(()=>void 0!==h,[h]),{direction:w,spin:j}=l.useContext(s.E_),z=o()(t,null==j?void 0:j.className,{[`${t}-sm`]:"small"===m,[`${t}-lg`]:"large"===m,[`${t}-spinning`]:y,[`${t}-show-text`]:!!p,[`${t}-rtl`]:"rtl"===w},c,d,b),C=o()(`${t}-container`,{[`${t}-blur`]:y}),_=(0,a.Z)(v,["indicator","prefixCls"]),N=Object.assign(Object.assign({},null==j?void 0:j.style),g),O=l.createElement("div",Object.assign({},_,{style:N,className:z,"aria-live":"polite","aria-busy":y}),function(e,t){let{indicator:n}=t,i=`${e}-dot`;return null===n?null:(0,r.l$)(n)?(0,r.Tm)(n,{className:o()(n.props.className,i)}):(0,r.l$)(x)?(0,r.Tm)(x,{className:o()(x.props.className,i)}):l.createElement("span",{className:o()(i,`${e}-dot-spin`)},l.createElement("i",{className:`${e}-dot-item`,key:1}),l.createElement("i",{className:`${e}-dot-item`,key:2}),l.createElement("i",{className:`${e}-dot-item`,key:3}),l.createElement("i",{className:`${e}-dot-item`,key:4}))}(t,e),p&&$?l.createElement("div",{className:`${t}-text`},p):null);return $?l.createElement("div",Object.assign({},_,{className:o()(`${t}-nested-loading`,u,b)}),y&&l.createElement("div",{key:"loading"},O),l.createElement("div",{className:C,key:"container"},h)):O},y=e=>{let{prefixCls:t}=e,{getPrefixCls:n}=l.useContext(s.E_),i=n("spin",t),[o,a]=b(i),r=Object.assign(Object.assign({},e),{spinPrefixCls:i,hashId:a});return o(l.createElement(v,Object.assign({},r)))};y.setDefaultIndicator=e=>{x=e};var S=y},48312:function(e,t,n){(window.__NEXT_P=window.__NEXT_P||[]).push(["/",function(){return n(57464)}])},48567:function(e,t,n){"use strict";n.d(t,{Z:function(){return p},A:function(){return m}});var i=n(85893),o=n(41468),a=n(51009);let l={proxyllm:{label:"Proxy LLM",icon:"/models/chatgpt.png"},"flan-t5-base":{label:"flan-t5-base",icon:"/models/google.png"},"vicuna-13b":{label:"vicuna-13b",icon:"/models/vicuna.jpeg"},"vicuna-7b":{label:"vicuna-7b",icon:"/models/vicuna.jpeg"},"vicuna-13b-v1.5":{label:"vicuna-13b-v1.5",icon:"/models/vicuna.jpeg"},"vicuna-7b-v1.5":{label:"vicuna-7b-v1.5",icon:"/models/vicuna.jpeg"},"codegen2-1b":{label:"codegen2-1B",icon:"/models/vicuna.jpeg"},"codet5p-2b":{label:"codet5p-2b",icon:"/models/vicuna.jpeg"},"chatglm-6b-int4":{label:"chatglm-6b-int4",icon:"/models/chatglm.png"},"chatglm-6b":{label:"chatglm-6b",icon:"/models/chatglm.png"},"chatglm2-6b":{label:"chatglm2-6b",icon:"/models/chatglm.png"},"chatglm2-6b-int4":{label:"chatglm2-6b-int4",icon:"/models/chatglm.png"},"guanaco-33b-merged":{label:"guanaco-33b-merged",icon:"/models/huggingface.svg"},"falcon-40b":{label:"falcon-40b",icon:"/models/falcon.jpeg"},"gorilla-7b":{label:"gorilla-7b",icon:"/models/gorilla.png"},"gptj-6b":{label:"ggml-gpt4all-j-v1.3-groovy.bin",icon:""},chatgpt_proxyllm:{label:"chatgpt_proxyllm",icon:"/models/chatgpt.png"},bard_proxyllm:{label:"bard_proxyllm",icon:"/models/bard.gif"},claude_proxyllm:{label:"claude_proxyllm",icon:"/models/claude.png"},wenxin_proxyllm:{label:"wenxin_proxyllm",icon:""},tongyi_proxyllm:{label:"tongyi_proxyllm",icon:"/models/qwen2.png"},zhipu_proxyllm:{label:"zhipu_proxyllm",icon:"/models/zhipu.png"},"llama-2-7b":{label:"Llama-2-7b-chat-hf",icon:"/models/llama.jpg"},"llama-2-13b":{label:"Llama-2-13b-chat-hf",icon:"/models/llama.jpg"},"llama-2-70b":{label:"Llama-2-70b-chat-hf",icon:"/models/llama.jpg"},"baichuan-13b":{label:"Baichuan-13B-Chat",icon:"/models/baichuan.png"},"baichuan-7b":{label:"baichuan-7b",icon:"/models/baichuan.png"},"baichuan2-7b":{label:"Baichuan2-7B-Chat",icon:"/models/baichuan.png"},"baichuan2-13b":{label:"Baichuan2-13B-Chat",icon:"/models/baichuan.png"},"wizardlm-13b":{label:"WizardLM-13B-V1.2",icon:"/models/wizardlm.png"},"llama-cpp":{label:"ggml-model-q4_0.bin",icon:"/models/huggingface.svg"},"internlm-7b":{label:"internlm-chat-7b-v1_1",icon:"/models/internlm.png"},"internlm-7b-8k":{label:"internlm-chat-7b-8k",icon:"/models/internlm.png"}};var r=n(25675),s=n.n(r),c=n(67294),d=n(67421);function m(e,t){var n;let{width:o,height:a}=t||{};return e?(0,i.jsx)(s(),{className:"rounded-full border border-gray-200 object-contain bg-white inline-block",width:o||24,height:a||24,src:(null===(n=l[e])||void 0===n?void 0:n.icon)||"/models/huggingface.svg",alt:"llm"}):null}var p=function(e){let{onChange:t}=e,{t:n}=(0,d.$G)(),{modelList:r,model:s}=(0,c.useContext)(o.p);return!r||r.length<=0?null:(0,i.jsx)(a.default,{value:s,placeholder:n("choose_model"),className:"w-52",onChange:e=>{null==t||t(e)},children:r.map(e=>{var t;return(0,i.jsx)(a.default.Option,{children:(0,i.jsxs)("div",{className:"flex items-center",children:[m(e),(0,i.jsx)("span",{className:"ml-2",children:(null===(t=l[e])||void 0===t?void 0:t.label)||e})]})},e)})})}},38954:function(e,t,n){"use strict";n.d(t,{Z:function(){return y}});var i=n(85893),o=n(27496),a=n(59566),l=n(71577),r=n(67294),s=n(2487),c=n(83062),d=n(2453),m=n(74627),p=n(39479),u=n(51009),g=n(58299),h=n(577),b=n(30119),f=n(67421);let x=e=>{let{data:t,loading:n,submit:o,close:a}=e,{t:l}=(0,f.$G)(),r=e=>()=>{o(e),a()};return(0,i.jsx)("div",{style:{maxHeight:400,overflow:"auto"},children:(0,i.jsx)(s.Z,{dataSource:null==t?void 0:t.data,loading:n,rowKey:e=>e.prompt_name,renderItem:e=>(0,i.jsx)(s.Z.Item,{onClick:r(e.content),children:(0,i.jsx)(c.Z,{title:e.content,children:(0,i.jsx)(s.Z.Item.Meta,{style:{cursor:"copy"},title:e.prompt_name,description:l("Prompt_Info_Scene")+"：".concat(e.chat_scene,"，")+l("Prompt_Info_Sub_Scene")+"：".concat(e.sub_chat_scene)})})},e.prompt_name)})})};var v=e=>{let{submit:t}=e,{t:n}=(0,f.$G)(),[o,a]=(0,r.useState)(!1),[l,s]=(0,r.useState)("common"),{data:v,loading:y}=(0,h.Z)(()=>(0,b.PR)("/prompt/list",{prompt_type:l}),{refreshDeps:[l],onError:e=>{d.ZP.error(null==e?void 0:e.message)}});return(0,i.jsx)(m.Z,{title:(0,i.jsx)(p.Z.Item,{label:"Prompt "+n("Type"),children:(0,i.jsx)(u.default,{style:{width:130},value:l,onChange:e=>{s(e)},options:[{label:n("Public")+" Prompts",value:"common"},{label:n("Private")+" Prompts",value:"private"}]})}),content:(0,i.jsx)(x,{data:v,loading:y,submit:t,close:()=>{a(!1)}}),placement:"topRight",trigger:"click",open:o,onOpenChange:e=>{a(e)},children:(0,i.jsx)(c.Z,{title:n("Click_Select")+" Prompt",children:(0,i.jsx)(g.Z,{className:"bottom-32"})})})},y=function(e){let{children:t,loading:n,onSubmit:s,...c}=e,[d,m]=(0,r.useState)("");return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(a.default.TextArea,{className:"flex-1",size:"large",value:d,autoSize:{minRows:1,maxRows:4},...c,onPressEnter:e=>{if(d.trim()&&13===e.keyCode){if(e.shiftKey){m(e=>e+"\n");return}s(d),setTimeout(()=>{m("")},0)}},onChange:e=>{if("number"==typeof c.maxLength){m(e.target.value.substring(0,c.maxLength));return}m(e.target.value)}}),(0,i.jsx)(l.ZP,{className:"ml-2 flex items-center justify-center",size:"large",type:"text",loading:n,icon:(0,i.jsx)(o.Z,{}),onClick:()=>{s(d)}}),(0,i.jsx)(v,{submit:e=>{m(d+e)}}),t]})}},57464:function(e,t,n){"use strict";n.r(t);var i=n(85893),o=n(577),a=n(67294),l=n(96074),r=n(75081),s=n(39332),c=n(25675),d=n.n(c),m=n(50489),p=n(48567),u=n(41468),g=n(38954),h=n(67421),b=n(8937);t.default=()=>{let e=(0,s.useRouter)(),{model:t,setModel:n}=(0,a.useContext)(u.p),{t:c}=(0,h.$G)(),[f,x]=(0,a.useState)(!1),[v,y]=(0,a.useState)(!1),{data:S=[]}=(0,o.Z)(async()=>{y(!0);let[,e]=await (0,m.Vx)((0,m.CU)());return y(!1),null!=e?e:[]}),$=async n=>{x(!0);let[,i]=await (0,m.Vx)((0,m.sW)({chat_mode:"chat_normal"}));i&&(localStorage.setItem(b.rU,JSON.stringify({id:i.conv_uid,message:n})),e.push("/chat/?scene=chat_normal&id=".concat(i.conv_uid).concat(t?"&model=".concat(t):""))),x(!1)},w=async n=>{let[,i]=await (0,m.Vx)((0,m.sW)({chat_mode:"chat_normal"}));i&&e.push("/chat?scene=".concat(n.chat_scene,"&id=").concat(i.conv_uid).concat(t?"&model=".concat(t):""))};return(0,i.jsxs)("div",{className:"mx-auto h-full justify-center flex max-w-3xl flex-col px-4",children:[(0,i.jsx)("div",{className:"my-0 mx-auto",children:(0,i.jsx)(d(),{src:"/LOGO.png",alt:"Revolutionizing Database Interactions with Private LLM Technology",width:856,height:160,className:"w-full",unoptimized:!0})}),(0,i.jsx)(l.Z,{className:"!text-[#878c93] !my-6",plain:!0,children:c("Quick_Start")}),(0,i.jsx)(r.Z,{spinning:v,children:(0,i.jsx)("div",{className:"flex flex-wrap -m-1 md:-m-3",children:S.map(e=>(0,i.jsx)("div",{className:"w-full sm:w-1/2 lg:w-1/3 p-1 md:p-3",children:(0,i.jsx)("div",{className:"cursor-pointer flex items-center justify-center w-full h-12 rounded font-semibold text-sm bg-[#E6F4FF] text-[#1677FE] dark:text-gray-100 dark:bg-[#4E4F56]",onClick:()=>{w(e)},children:e.scene_name})},e.chat_scene))})}),(0,i.jsx)("div",{className:"mt-8 mb-2",children:(0,i.jsx)(p.Z,{onChange:e=>{n(e)}})}),(0,i.jsx)("div",{className:"flex",children:(0,i.jsx)(g.Z,{loading:f,onSubmit:$})})]})}},30119:function(e,t,n){"use strict";n.d(t,{Tk:function(){return s},PR:function(){return c}});var i=n(2453),o=n(6154),a=n(83454);let l=o.Z.create({baseURL:a.env.API_BASE_URL});l.defaults.timeout=1e4,l.interceptors.response.use(e=>e.data,e=>Promise.reject(e)),n(96486);let r={"content-type":"application/json"},s=(e,t)=>{if(t){let n=Object.keys(t).filter(e=>void 0!==t[e]&&""!==t[e]).map(e=>"".concat(e,"=").concat(t[e])).join("&");n&&(e+="?".concat(n))}return l.get("/api"+e,{headers:r}).then(e=>e).catch(e=>{i.ZP.error(e),Promise.reject(e)})},c=(e,t)=>l.post(e,t,{headers:r}).then(e=>e).catch(e=>{i.ZP.error(e),Promise.reject(e)})}},function(e){e.O(0,[662,44,479,9,411,270,539,774,888,179],function(){return e(e.s=48312)}),_N_E=e.O()}]);