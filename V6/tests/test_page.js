// Exercise the actual inline page script with a tiny DOM and local API stub.
// Built-in Node modules only; no browser, model or screen capture is started.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../static/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

class Element {
  constructor(tagName = '') { this.tagName = tagName; this.hidden = false; this.disabled = false; this.textContent = ''; this.events = {}; this.attributes = {}; }
  addEventListener(name, callback) { this.events[name] = callback; }
  replaceChildren(...children) { this.children = children; }
  append(...children) { this.children = children; }
  appendChild(child) { (this.children ||= []).push(child); }
  setAttribute(name, value) { this.attributes[name] = value; }
}
const settle = async () => { for (let i = 0; i < 20; i++) await new Promise(setImmediate); };
function page(initial) {
  const elements = new Map();
  const timers = new Map();
  let timerId = 0;
  const api = {state: initial, offline: false, resultFailure: false, timelineFailure: false, timeline: [], calls: [],
    result: {verdict: 'AUTHENTIC / REAL', scores: {average: .1, peak: .2, burst: .15, wci: .14},
      debug: {frames_captured: 12, frames_with_detection: 8, usable_crops_after_filters: 6}}};
  const get = (id) => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  vm.runInNewContext(script, {
    document: {getElementById: get, createElement: (tag) => new Element(tag), createElementNS: (_, tag) => new Element(tag)},
    AbortController,
    setTimeout: (callback, delay) => { const id = ++timerId; timers.set(id, {callback, delay}); return id; },
    clearTimeout: (id) => timers.delete(id),
    fetch: async (url, options) => {
      api.calls.push([url, options.method]);
      assert(url.startsWith('/api/live/'), 'Only local backend API requests are allowed');
      if (api.offline) throw new TypeError('offline');
      let body = api.state, ok = true;
      if (url.endsWith('/start')) {
        api.state = {session_id: 'new-session', status: 'starting', process_alive: true};
        body = {session_id: 'new-session'};
      } else if (url.endsWith('/stop')) {
        api.state = {...api.state, status: 'done', process_alive: false, stop_requested: true};
        body = {stop_requested: true};
      } else if (url.includes('/timeline/')) {
        body = api.timelineFailure ? {detail: 'Timeline unreadable'} : {samples: api.timeline};
        ok = !api.timelineFailure;
      } else if (url.includes('/result/')) {
        body = api.resultFailure ? {detail: 'Result file missing'} : api.result;
        ok = !api.resultFailure;
      }
      return {ok, status: ok ? 200 : 404, json: async () => body};
    }
  });
  return {api, get, async poll() {
    const entry = [...timers].find(([, timer]) => timer.delay === 1000);
    assert(entry, 'A one-second status poll must be scheduled');
    timers.delete(entry[0]);
    await entry[1].callback();
    await settle();
  }};
}

(async () => {
  const p = page({session_id: null, status: 'idle', process_alive: false});
  await settle();
  assert.equal(p.get('start').disabled, false);
  assert.equal(p.get('stop').disabled, true);
  await p.get('start').events.click(); await settle();
  assert.equal(p.get('start').disabled, true);
  assert.equal(p.get('stop').disabled, false);
  await p.get('stop').events.click(); await settle();
  assert.equal(p.get('verdict').textContent, 'REAL');
  assert.equal(p.get('pdf').href, '/api/live/report/new-session');
  assert.equal(p.get('counts').children[2].children[1].textContent, '6');
  assert.equal(p.get('start').disabled, false);
  assert.equal(p.get('stop').disabled, true);

  const restored = page({session_id: 'existing-session', status: 'capturing', process_alive: true});
  await settle();
  assert.equal(restored.get('start').disabled, true, 'Reload must pick up a live session');
  assert.equal(restored.get('stop').disabled, false);
  assert(!restored.api.calls.some(([url]) => url.endsWith('/start')), 'Reload must not start another runner');
  for (const stage of ['countdown', 'selecting_region', 'analyzing']) {
    restored.api.state.status = stage; await restored.poll();
    assert(restored.get('status-text').textContent.length > 10);
  }
  restored.api.state = {session_id: 'existing-session', status: 'capturing', process_alive: false};
  await restored.poll();
  assert.equal(restored.get('error-title').textContent, 'Analysis failed');
  assert.equal(restored.get('start').disabled, false);
  restored.api.state.status = 'cancelled'; await restored.poll();
  assert.equal(restored.get('error-title').textContent, 'Analysis cancelled');
  assert.equal(restored.get('start').disabled, false);

  const result = page({session_id: 'fake', status: 'done', process_alive: false});
  result.api.result.verdict = 'SYNTHETIC / DEEPFAKE';
  await settle();
  assert.equal(result.get('verdict').textContent, 'DEEPFAKE');
  result.api.state.session_id = 'empty'; result.api.result.verdict = 'Not enough evidence';
  await result.poll();
  assert.equal(result.get('verdict').textContent, 'Not enough evidence');
  assert.equal(result.get('metrics').children[0].children[1].textContent, '—');

  const failed = page({session_id: 'missing-result', status: 'done', process_alive: false});
  failed.api.resultFailure = true; await settle();
  assert.equal(failed.get('start').disabled, false);
  assert(failed.get('status-text').textContent.includes('could not be loaded'));
  failed.api.offline = true; await failed.poll();
  assert.equal(failed.get('error-title').textContent, 'Backend not running');
  assert.equal(failed.get('start-command').hidden, false);
  failed.api.offline = false; failed.api.state = {session_id: null, status: 'idle', process_alive: false};
  await failed.poll();
  assert.equal(failed.get('start').disabled, false);

  const chart = page({session_id: 'one-point', status: 'done', process_alive: false});
  chart.api.timeline = [{elapsed_seconds: 1.25, adjusted_fake_score: 0.8, timestamp: 'recorded time'}];
  chart.api.result.verdict_explanation = 'Recorded evidence: 6 usable crops. This does not prove authenticity.';
  await settle();
  assert.equal(chart.get('summary').textContent, chart.api.result.verdict_explanation);
  assert(chart.get('timeline-message').textContent.includes('1 recorded samples'));
  assert.equal(chart.get('timeline-legend').hidden, false);
  assert.equal(chart.get('timeline-svg').children.filter(n => n.tagName === 'circle').length, 1);
  assert.equal(chart.get('timeline-svg').children.find(n => n.tagName === 'circle').attributes.fill, '#6d4dbf');

  chart.api.state.session_id = 'multiple';
  chart.api.timeline = [null, {elapsed_seconds: -1, adjusted_fake_score: .99},
    {elapsed_seconds: 120, adjusted_fake_score: .81}, {elapsed_seconds: 0, adjusted_fake_score: .80}];
  await chart.poll();
  const dots = chart.get('timeline-svg').children.filter(n => n.tagName === 'circle');
  assert.deepEqual(dots.map(n => n.attributes.fill), ['#6d4dbf', '#a83a50']);
  assert.equal(chart.get('timeline-end').textContent, '2.0 min');
  assert(chart.get('timeline-svg').children.some(n => n.textContent === 'Elapsed time (min)'));

  chart.api.state.session_id = 'empty-timeline'; chart.api.timeline = [];
  await chart.poll();
  assert.equal(chart.get('timeline-start').textContent, '—');
  assert.equal(chart.get('timeline-end').textContent, '—');
  assert.equal(chart.get('timeline-legend').hidden, true);
  assert.equal(chart.get('timeline-svg').children.length, 0);
  assert(chart.get('timeline-message').textContent.includes('No usable timed'));

  chart.api.state.session_id = 'failed-timeline'; chart.api.timelineFailure = true;
  await chart.poll();
  assert.equal(chart.get('timeline-end').textContent, '—');
  assert(chart.get('timeline-message').textContent.includes('could not be loaded'));
  assert.equal(chart.get('pdf').hidden, false, 'A failed chart must not hide the report');

  chart.api.state.session_id = 'large'; chart.api.timelineFailure = false;
  chart.api.timeline = Array.from({length: 200000}, (_, i) => ({elapsed_seconds: i / 30, adjusted_fake_score: .1}));
  chart.api.timeline[12345].adjusted_fake_score = .99;
  await chart.poll();
  const largeDots = chart.get('timeline-svg').children.filter(n => n.tagName === 'circle');
  assert(largeDots.length <= 800);
  assert(largeDots.some(n => n.attributes.fill === '#a83a50'), 'An isolated peak must survive display reduction');
  assert(chart.get('timeline-message').textContent.includes('200000 recorded samples; 1'));
  console.log('PASS: live controls, verdicts/explanations, reload/error recovery; empty/one/multiple/200000-point charts, numeric ticks and display-only highlights.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
